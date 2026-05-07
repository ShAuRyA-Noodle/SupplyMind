"""DPO-fine-tune a 3B Qwen as a calibrated SupplyMind risk judge.

Entry points:
    prepare_preference_data:  26 crisis scenarios -> (prompt, chosen, rejected) triples
    train_dpo_trl:            standalone DPO via HuggingFace trl (ROLL-free fallback)
    train_dpo_roll:           DPO via ROLL pipeline (if ROLL installed)
"""
