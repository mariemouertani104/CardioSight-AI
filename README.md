# CardioSight-AI
J'ai développé un système ECG dual : un classificateur d'arythmies en temps réel ET un système d'authentification biométrique basé sur le signal cardiaque. Les deux utilisent le même pipeline de traitement de signaux physiologiques continus
# 🫀 ECG Intelligence Platform

> **Projet d'Ingénierie IA Médicale** — Classification des Arythmies & Identification Biométrique ECG  
> Développé en 4 jours | Google Colab | Python · TensorFlow · Gemini 2.5 Flash

---

##  Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Résultats Clés](#résultats-clés)
3. [Architecture du Système](#architecture-du-système)
4. [Projet 1 — Classification des Arythmies](#projet-1--classification-des-arythmies)
5. [Projet 2 — Identification Biométrique ECG](#projet-2--identification-biométrique-ecg)
6. [AI Agent — Génération de Rapports PDF](#ai-agent--génération-de-rapports-pdf)
7. [Défis Techniques & Solutions](#défis-techniques--solutions)
8. [Installation & Utilisation](#installation--utilisation)
9. [Connexion avec applicare.ai](#connexion-avec-applicareai)
10. [Références](#références)

---

##  Vue d'ensemble

Ce projet implémente un **système dual d'analyse ECG par intelligence artificielle**, composé de deux modules de Deep Learning et d'un agent IA génératif :

| Module | Objectif | Technologie |
|--------|----------|-------------|
| **Classification** | Détecter 5 types d'arythmies cardiaques | ResNet1D + SMOTE + Focal Loss |
| **Biométrie** | Identifier un patient par son signal ECG | BiometricNet CNN |
| **AI Agent** | Générer rapports médecin + patient en PDF | Gemini 2.5 Flash |

**Contexte** : Ce projet démontre des compétences en IA médicale appliquée.

>  **Datasets utilisés** :
> - MIT-BIH Arrhythmia Database (Kaggle — `shayanfazeli/heartbeat`) pour la classification
> - MIT-BIH PhysioNet (via `wfdb`) pour la biométrie — 47 vrais patients

<img width="1306" height="694" alt="image" src="https://github.com/user-attachments/assets/698b61dd-8371-4f8c-95ce-9c89ef6ea685" />

---

##  Résultats Clés

### Projet 1 — Classification des Arythmies

| Métrique | Modèle Initial | Modèle Final |
|----------|---------------|--------------|
| **Accuracy** | 82.76% (tout Normal) | **98.58%** |
| **Macro F1** | 0.18 | **0.9259** |
| **AUC moyen** | — | **0.9957** |

| Classe | F1-Score | AUC | Statut |
|--------|----------|-----|--------|
| N — Normal | 0.9927 | 0.9967 | ✅ Excellent |
| S — Supraventriculaire | 0.8273 | 0.9839 | ✅ Bon |
| V — Ventriculaire | 0.9673 | 0.9984 | ✅ Excellent |
| F — Fusion | 0.8504 | 0.9934 | ✅ Excellent |
| Q — Indéterminé | 0.9919 | 0.9999 | ✅ Excellent |

<img width="1458" height="614" alt="image" src="https://github.com/user-attachments/assets/c9be31d9-e95c-406e-9c82-b334257f37aa" />
<img width="1458" height="466" alt="image" src="https://github.com/user-attachments/assets/e1f7ff4a-f19f-4880-92a8-ffc77283c09e" />
<img width="530" height="286" alt="image" src="https://github.com/user-attachments/assets/ffc596ee-fbf8-4526-b545-0276ab21740b" />





### Projet 2 — Identification Biométrique ECG

| Métrique | Valeur |
|----------|--------|
| **Accuracy Top-1** | **92.35%** |
| **Accuracy Top-3** | **96.26%** |
| Précision (macro) | 0.9284 |
| Rappel (macro) | 0.9228 |
| F1-Score (macro) | 0.9222 |
| **Verdict** |  **EXCELLENT** |
<img width="471" height="190" alt="image" src="https://github.com/user-attachments/assets/021643cc-c2f2-4b8b-b1fb-6a4220203f42" />
<img width="1414" height="591" alt="image" src="https://github.com/user-attachments/assets/0b79d4b5-02e5-4f67-9ad2-a1c4c162b9cb" />
<img width="1491" height="464" alt="image" src="https://github.com/user-attachments/assets/c20781a3-dda5-4df1-993f-48f49b429768" />
<img width="1374" height="664" alt="image" src="https://github.com/user-attachments/assets/1725933f-bdcd-428b-8119-4b4acabaa5b1" />


---

##  Architecture du Système

```
ECG Intelligence Platform
│
├── 📁 Module 1 : Classification des Arythmies
│   ├── Dataset       : MIT-BIH Kaggle — 109,446 battements, 5 classes
│   ├── Prétraitement : StandardScaler → Undersampling → SMOTE
│   ├── Modèle        : ResNet1D (blocs résiduels) + Focal Loss
│   ├── Post-traitement : Optimisation seuils par classe (grid search F1)
│   └── Évaluation    : ROC, PR Curves, Confusion Matrix, KPIs médicaux
│
├── 📁 Module 2 : Identification Biométrique
│   ├── Dataset       : PhysioNet MIT-BIH — 47 vrais patients (wfdb)
│   ├── Prétraitement : Filtre Butterworth 0.5–40 Hz + extraction battements PQRST
│   ├── Modèle        : BiometricNet CNN + Data Augmentation ×4
│   └── Évaluation    : Top-1/Top-3, PCA Embeddings, F1 par patient
│
└── 📁 Module 3 : AI Agent
    ├── LLM           : Google Gemini 2.5 Flash
    ├── Rapport 1     : Rapport médical professionnel (PDF téléchargeable)
    └── Rapport 2     : Explication patient simplifiée (PDF téléchargeable)
```

---

## 🔬 Projet 1 — Classification des Arythmies

### Le Problème du Déséquilibre de Classes

```
Classe N (Normal)          : 72,471 échantillons  ████████████████████ 82.77%
Classe S (Supraventriculaire):  2,223 échantillons  █ 2.54%
Classe V (Ventriculaire)   :  5,788 échantillons  ██ 6.61%
Classe F (Fusion)          :    641 échantillons  ▌ 0.73%
Classe Q (Indéterminé)     :  6,431 échantillons  ██ 7.34%
```

Sans correction → le modèle CNN basique prédit **tout comme "Normal"** → **0% de détection des arythmies** → résultat cliniquement inutile.

### Pipeline de Résolution




**Étape 1 — Undersampling**
- Classe Normal réduite de 72,471 → 10,000 échantillons
  <img width="879" height="115" alt="image" src="https://github.com/user-attachments/assets/08309a10-f5d0-43a4-b1be-c71de25531cb" />


**Étape 2 — SMOTE**
- Classes minoritaires augmentées synthétiquement → 10,000 chacune

**Étape 3 — Focal Loss**
```python
# Pénalise plus les erreurs sur les classes rares
focal_loss(gamma=2.0, alpha=0.25)
```

**Étape 4 — Optimisation des seuils de décision**
```python
# Grid search sur 50 seuils entre 0.1 et 0.9 par classe
# Optimisation du F1-Score par classe
best_thresholds = {N: 0.100, S: 0.835, V: 0.655, F: 0.900, Q: 0.884}
```

### Architecture ResNet1D

```
Input (187, 1)
    │
    ├── Conv1D(64, k=7) + BatchNorm + MaxPool
    │
    ├── Residual Block 1 (64 filters)
    │   └── Conv→BN→ReLU→Conv→BN + Skip Connection
    │
    ├── Residual Block 2 (128 filters)
    │
    ├── Residual Block 3 (256 filters)
    │
    ├── GlobalAveragePooling1D
    │
    ├── Dense(128) + BatchNorm + Dropout(0.4)
    ├── Dense(64) + Dropout(0.3)
    └── Dense(5, softmax)
```

### Progression des Performances

| Étape | Modèle | Accuracy | S F1 | V F1 | Macro F1 |
|-------|--------|----------|------|------|----------|
| 1 | CNN basique | 82.76% | 0.00 | 0.00 | 0.18 |
| 2 | CNN + Undersampling | 97.27% | 0.71 | 0.94 | 0.87 |
| 3 | ResNet1D + SMOTE | 97.27% | 0.72 | 0.94 | 0.87 |
| **4** | **+ Focal Loss + Seuils** | **98.58%** | **0.83** | **0.97** | **0.93** |

---

##  Projet 2 — Identification Biométrique ECG

### Concept

> *"Votre rythme cardiaque comme mot de passe biométrique — personne ne peut voler votre cœur."*

Chaque individu possède une **signature ECG unique** liée à l'anatomie de son cœur (taille, position, morphologie). Ce module apprend à reconnaître cette empreinte parmi 47 patients réels.

### Pourquoi les "faux patients" CSV ont échoué

Première tentative : découper le CSV MIT-BIH en 15 blocs → **15% accuracy** (quasi-aléatoire).
- Raison : les battements anonymisés n'ont aucune identité individuelle réelle
- Solution : utiliser les **vrais enregistrements wfdb** de PhysioNet avec les IDs patients réels

### Pipeline d'Extraction

```python
# 1. Téléchargement des 48 records MIT-BIH via PhysioNet
record = wfdb.rdrecord(record_name, pn_dir='mitdb', channels=[0])

# 2. Filtrage Butterworth passe-bande
b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
signal_filtered = filtfilt(b, a, signal)

# 3. Extraction des battements autour des pics R annotés
# Fenêtre : 180 points (±90 autour du pic R)
beat = signal[sample-90 : sample+90]

# 4. Normalisation min-max par battement
beat = (beat - beat.min()) / (beat.max() - beat.min())

# 5. Augmentation ×4 (bruit gaussien + scaling)
X_aug, y_aug = augment_ecg(X_train, y_train, factor=4)
```

### Architecture BiometricNet

```
Input (180, 1)
    │
    ├── Conv1D(64, k=5) + BatchNorm + MaxPool + Dropout(0.1)
    ├── Conv1D(128, k=5) + BatchNorm + MaxPool + Dropout(0.1)
    ├── Conv1D(256, k=3) + BatchNorm + MaxPool + Dropout(0.2)
    │
    ├── GlobalAveragePooling1D
    │
    ├── Dense(128) + BatchNorm + Dropout(0.3)
    ├── Dense(64)
    └── Dense(47, softmax)  ← 47 patients
```

### Résultat Démonstration Live

```
DEMO LIVE — Identification Patient Inconnu
==========================================
Patient réel      : P101
Patient identifié : P101
Confiance         : 100.0%
Résultat          : ✅ CORRECT

Top-3 :
  1. P101 : 100.0% ← CORRECT
  2. P231 :   0.0%
  3. P102 :   0.0%
```

---

## 🤖 AI Agent — Génération de Rapports PDF

### Architecture de l'Agent

```
Résultats Classification (98.58% acc.)
+
Résultats Biométrie (92.35% Top-1)
            ↓
    [Gemini 2.5 Flash]
       ↙           ↘
Rapport Médecin   Rapport Patient
(Terminologie     (Langage simple,
 médicale,         rassurant,
 recommandations,  analogies,
 niveau urgence)   sans jargon)
       ↓               ↓
   PDF auto-        PDF auto-
   téléchargé       téléchargé
```

### Contenu des Rapports

**Rapport Médecin** inclut :
1. Résultat principal (diagnostic + confiance)
2. Paramètres cardiaques (FC, rythme, probabilités)
3. Niveau d'urgence (VERT / ORANGE / ROUGE)
4. Recommandations cliniques
5. Identification biométrique du patient
6. Limites du système IA

**Rapport Patient** inclut :
- Explication simple avec analogies
- Fréquence cardiaque vulgarisée
- Actions concrètes selon le risque
- Message rassurant et humain

### Clause Éthique Systématique

> *" Ce rapport est généré automatiquement par un système IA.  
> Il constitue une aide à la décision et ne remplace pas le jugement  
> d'un professionnel de santé qualifié."*

---

## 🛠️ Défis Techniques & Solutions

| Défi Rencontré | Impact | Solution Adoptée | Résultat |
|----------------|--------|-----------------|----------|
| Déséquilibre 82.77% Normal | 0% détection arythmies | Undersampling + SMOTE + Focal Loss | 98.58% accuracy |
| APC montre chevauchement (52%) | Justifie CNN non-linéaire | Architecture ResNet1D résiduelle | AUC > 0.99 |
| Lambda layers Keras 3 incompatibles | Erreur InvalidArgumentError | Couches custom `L2Distance`, `L2Normalize` | Résolu |
| Multi-input Keras 3 cassé | Siamese Network bloqué | Single-input concaténé (374 pts) | Résolu |
| Seuil 0.5 universel | Classe S sous-performante | Grid search seuil optimal par classe | S: 0.71→0.83 |

---

##  Installation & Utilisation

### Environnement

```bash
# Google Colab (recommandé)
!pip install tensorflow scikit-learn imbalanced-learn
!pip install wfdb scipy matplotlib seaborn pandas numpy
!pip install google-generativeai reportlab joblib kaggle
```

### Configuration

```python
# 1. Clé Gemini dans les Secrets Colab
#     Panneau gauche → Secrets → GEMINI_API_KEY → votre_clé

# 2. Dataset MIT-BIH depuis Kaggle
!kaggle datasets download -d shayanfazeli/heartbeat
!unzip heartbeat.zip -d heartbeat_data/

# 3. Exécuter les sections dans l'ordre :
#    Partie A : Classification (Projet 1) ── ~15 min d'entraînement
#    Partie B : Biométrie (Projet 6)     ── ~10 min d'entraînement
#    Partie C : AI Agent + PDF           ── ~30 secondes
```

### Fichiers Sauvegardés

```
models/
├── best_ecg_model.keras       ← Classifieur ResNet1D (98.58%)
├── ecg_biometric_net.keras    ← BiometricNet (92.35% Top-1)
├── biometric_config.pkl       ← Configuration 47 patients
└── scaler.pkl                 ← StandardScaler fitted

outputs/
├── dashboard_complet.png      ← KPIs + Confusion + ROC + PR + Risk
├── biometric_final_dashboard.png
├── rapport_medecin_*.pdf      ← Rapport généré par Gemini
└── rapport_patient_*.pdf
```

---


##  Références

- Moody G.B., Mark R.G. (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology, 20(3):45-50.
- Goldberger A., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23).
- Lin, T.Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
- Chawla N.V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16:321-357.
- He K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.

---

##  Auteure

**Ouertani Mariem Étudiante Ingénieure — Network & Telecommunications**  
Spécialisation : AI · ML · Deep Learning · Signal Processing · Data Analytics

**Stack technique** : Python · TensorFlow/Keras · scikit-learn · SciPy · Pandas · NumPy  
**Domaines couverts** : Traitement du signal ECG · Classification multi-classes · Biométrie · LLM Agents · IA médicale

---

*Projet réalisé avec Google Colab (GPU T4 gratuit) — Avril 2026*  
*Durée de développement : 4 jours*
