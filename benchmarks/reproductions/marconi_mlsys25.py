"""Synthetic Marconi admission/eviction benchmark."""

from vllm.v1.core.marconi_policy import MarconiEntry, MarconiPolicy, MarconiPolicyConfig


def main() -> None:
    entries = [
        MarconiEntry("attention", 8 * 1024**2, 2_000, 10, 0.6),
        MarconiEntry("recurrent", 512 * 1024, 1_000, 4, 0.8),
        MarconiEntry("cold", 16 * 1024**2, 500, 1, 0.1),
    ]
    policy = MarconiPolicy(MarconiPolicyConfig(efficiency_weight=2.0))
    for entry in entries:
        print(entry.key, f"admission_score={policy.admission_score(entry):.6g}")
    victims = policy.choose_victims(entries, bytes_to_free=8 * 1024**2, now=10)
    print("victims:", [entry.key for entry in victims])


if __name__ == "__main__":
    main()
