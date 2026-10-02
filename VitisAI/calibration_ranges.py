"""Escolhe faixas por camada usando apenas os dados de calibracao."""
import torch
from copy import deepcopy

POSITIONS = list(range(-12, 13))


class FixedRangeHistory(list):
    def append(self, position):
        pass  # Faixa ja escolhida pela observacao de todas as amostras.


class LayerRanges:
    def __init__(self, config, max_clipping_percent=0.1, samples_per_layer=2048):
        self.config = config
        self.max_clipping_percent = max_clipping_percent
        self.stats = {}
        self.samples_per_layer = samples_per_layer

    def capture(self, module, inputs, output):
        node = getattr(module, "node", None)
        if node is None or node.name not in self.config["output"] or not isinstance(output, torch.Tensor):
            return
        # ponytail: amostra uniforme limitada por camada/batch;
        # usar histogramas completos se a estimativa de saturacao for insuficiente.
        flat = output.detach().flatten()
        if not torch.isfinite(flat).all():
            raise ValueError(f"Ativacoes nao finitas em {node.name}")
        sample = flat[torch.linspace(0, flat.numel() - 1, min(self.samples_per_layer, flat.numel()), device=flat.device).long()]
        scales = torch.tensor([2. ** p for p in POSITIONS], device=sample.device).unsqueeze(1)
        bits = self.config["output"][node.name][0][0]
        low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        scaled = sample.unsqueeze(0) * scales
        quantized = torch.floor(scaled + .5).clamp(low, high) / scales
        error = ((quantized - sample) ** 2).sum(1).cpu().double()
        clipped = ((scaled < low) | (scaled > high)).sum(1).cpu()
        stats = self.stats.setdefault(node.name, dict(count=0, error=torch.zeros(len(POSITIONS), dtype=torch.double),
                                                    clipped=torch.zeros(len(POSITIONS), dtype=torch.long), max_abs=0., energy=0.))
        stats["count"] += sample.numel()
        stats["error"] += error
        stats["clipped"] += clipped
        stats["max_abs"] = max(stats["max_abs"], float(flat.abs().max()))
        stats["energy"] += float(sample.double().square().sum())

    def apply(self, quantizer):
        if not self.stats:
            raise RuntimeError("Nenhuma camada foi observada; verificar interface do Vitis AI.")
        for name, stats in self.stats.items():
            percentages = 100 * stats["clipped"].double() / stats["count"]
            valid = percentages <= self.max_clipping_percent
            if not valid.any():
                raise ValueError(f"Nenhuma faixa cobre as ativacoes de {name}")
            errors = stats["error"].clone()
            errors[~valid] = float("inf")
            index = int(errors.argmin())
            position = POSITIONS[index]
            bits = self.config["output"][name][0][0]
            quantizer.set_quant_config(name, [bits, position], "output", 0)
            quantizer.config_history["output"][name][0] = FixedRangeHistory([position])
            stats["position"] = position

    def refine(self, quantizer, model, reference, loader, products, device, layer_count):
        """Busca local nos logits de calibracao, projetando candidatos nas regras DPU."""
        from common import DataNormalizer
        dataset = loader.dataset
        records = getattr(dataset, "records", None)
        groups = [[i for i, record in enumerate(records) if record[2] == group]
                  for group in ("strong", "weak", "background")] if records else [list(range(len(dataset))) ]
        indices = [group[i] for group in groups for i in
                   torch.linspace(0, len(group) - 1, min(24 // len(groups), len(group))).long().tolist()]
        normalizer = DataNormalizer(products).to(device).eval()
        samples = []
        with torch.no_grad():
            for index in indices:
                raw = dataset[index]["input"]
                if raw.ndim == 4:
                    raw = raw[len(raw) // 2]
                x = normalizer(raw.unsqueeze(0).to(device))
                samples.append((x, reference(x).detach()))

        def restore(config):
            quantizer.quant_config.clear()
            quantizer.quant_config.update(deepcopy(config))

        def project():
            # ponytail: API interna Vitis AI 3.5; revisar ao trocar o quantizador.
            for _ in range(4):
                before = deepcopy(quantizer.quant_config)
                quantizer.organize_quant_pos()
                if before == quantizer.quant_config:
                    return True
            return False

        def score():
            with torch.no_grad():
                value = sum(float((model(x) - y).square().mean() / y.square().mean().clamp_min(1e-8))
                            for x, y in samples) / len(samples)
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("Erro nao finito durante refinamento")
            return value

        if not project():
            raise RuntimeError("Ajustes DPU nao estabilizaram antes do refinamento")
        best = deepcopy(quantizer.quant_config)
        baseline = best_score = score()
        positions = {name: best["output"][name][0][1] for name in self.stats}
        caps = {name: max(self.max_clipping_percent,
                         float(100 * stats["clipped"][POSITIONS.index(positions[name])] / stats["count"]))
                for name, stats in self.stats.items()}
        ranked = sorted(self.stats, key=lambda name:
                        float(self.stats[name]["error"][POSITIONS.index(positions[name])]) /
                        max(self.stats[name]["energy"], 1e-8), reverse=True)[:layer_count]
        trials = []
        # ponytail: 24 patches de calibracao e uma passagem pelas camadas de maior
        # erro relativo; ampliar apenas se a validacao justificar o custo.
        for name in ranked:
            center = best["output"][name][0][1]
            for candidate in (center - 1, center + 1, center - 2, center + 2):
                if candidate not in POSITIONS:
                    continue
                restore(best)
                bits = best["output"][name][0][0]
                quantizer.set_quant_config(name, [bits, candidate], "output", 0)
                if not project() or quantizer.quant_config["param"] != best["param"]:
                    continue
                current = quantizer.quant_config
                if current == best:
                    continue
                feasible = True
                for layer, stats in self.stats.items():
                    position = current["output"][layer][0][1]
                    if position not in POSITIONS or float(100 * stats["clipped"][POSITIONS.index(position)] / stats["count"]) > caps[layer] + 1e-7:
                        feasible = False
                        break
                if not feasible:
                    continue
                error = score()
                accepted = error < best_score - 1e-10
                trials.append(dict(layer=name, candidate=candidate, error=error, accepted=accepted))
                if accepted:
                    best, best_score = deepcopy(current), error
            restore(best)
            print(f"Refinamento: {name}; erro relativo dos logits={best_score:.6g}", flush=True)
        restore(best)
        return dict(baseline_error=baseline, final_error=best_score, metric="mean_per_patch_relative_logit_mse",
                    dataset_indices=indices, records=[records[i] for i in indices] if records else None,
                    layers_considered=ranked, trials=trials,
                    changes={name: [positions[name], best["output"][name][0][1]] for name in self.stats
                             if positions[name] != best["output"][name][0][1]})

    def report(self, quantizer):
        rows = []
        for name, stats in self.stats.items():
            final = quantizer.get_quant_config(name, False, "output", 0)[1]
            index = POSITIONS.index(final) if final in POSITIONS else None
            rows.append(dict(layer=name, selected_position=stats["position"], final_position=final,
                             sampled_values=stats["count"], max_abs=stats["max_abs"],
                             estimated_clipping_percent=None if index is None else float(100 * stats["clipped"][index] / stats["count"]),
                             estimated_mse=None if index is None else float(stats["error"][index] / stats["count"])))
        return rows
