# Pokémon Battle Prediction (FDS Kaggle Competition)

An XGBoost model to predict Pokémon battle winners based on in-game events, achieving 84.4% CV accuracy and a score of 0.8308 on the Private test set. 

**Authors:** Daniele Bracoloni, Marco Foresi, Alice Paglia  
**Group Name:** WolfeyVGC Academy Group  

---

This repository contains the code and methodology for the "FDS - Pokémon Battles Prediction" Kaggle competition. The project was developed as part of the Foundations of Data Science examination.

The objective of this project is to build a machine learning model capable of predicting the outcome of a battle between two Pokémon teams by analyzing both static team compositions and dynamic turn-by-turn battle timelines.

## Dataset

The data originates from the Kaggle competition: [FDS - Pokémon Battles Prediction 2025](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025). 

The dataset includes:
*   Pokémon attributes (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Types).
*   Detailed battle logs (timelines) recording every turn, move, and status change.
*   Historical battle records indicating the winner.

## Feature Engineering

While standard baselines rely solely on static base stats, this project implements a complex timeline-processing engine to simulate the battle and extract over 40 dynamic momentum features. 

Key feature engineering steps include:
*   **Temporal Checkpoints:** The state of the battle (HP, defeated Pokémon, and status conditions) is captured specifically at Turn 10 and Turn 20 to track momentum over time.
*   **Status Condition Stratification:** Status effects are split into `crucial_status` (Sleep, Freeze) and `minor_status` (Paralyze, Poison, Burn) to weigh their impact accurately.
*   **Type-Effectiveness Engine:** A full Gen-1 type matchup matrix dynamically counts `super_effective`, `resisted`, and `immune` hits throughout the match.
*   **Meta-Threat Detection:** The presence of top-tier Gen-1 competitive threats (Snorlax, Tauros, Chansey) is explicitly flagged to compute a `threat_advantage` metric.
*   **Momentum Metrics:** Offensive pressure is calculated via `avg_power` (with custom logic to properly handle fixed-damage moves like "Seismic Toss"), and lost turns are tracked via `null_moves`.
*   **Advantage & Ratio Features:** Engineered specific difference and ratio columns (e.g., `hp_ratio`, `first_ko_advantage`, `hp_adv_t10`) to provide direct relational data to the model.

## Modeling Strategy

The predictive model is built using `XGBClassifier`. To ensure robust generalization and prevent overfitting, a two-step validation pipeline was implemented:

1.  **Cross-Validation:** A 5-fold Stratified Cross-Validation (`StratifiedKFold`) was executed on the full training set to establish a highly reliable baseline accuracy.
2.  **Optimal Parameter Search:** A separate 85/15 `train_test_split` utilized `early_stopping_rounds` to identify the exact optimal number of trees (`best_iteration`).
3.  **Final Training:** The final XGBoost model was trained on 100% of the training data using the discovered optimal iteration count before predicting on the test set.

## Results

*   **Mean CV Accuracy (5-Fold):** 0.8444
*   **Private Test Score:** 0.8308 (Rank: 22)

## Repository Structure

*   `FDS_pokemon_challenge.ipynb`: The main Jupyter Notebook containing the end-to-end Pokédex construction, timeline parsing, feature engineering, and model training pipeline.
