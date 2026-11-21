---
title: FoodReviews AI
emoji: 🍽️
colorFrom: purple
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# FoodReviews AI : analyse et recommandation de produits

Démo interactive du projet *Advanced Learning for text and Graph Data* (Amazon Fine Food Reviews) :

- **Sentiment** d'un avis (DistilBERT fine-tuné, repli TF-IDF + régression logistique)
- **Recommandation** de produits (LightGCN + BPR sur le graphe utilisateurs-produits)
- **Q&A** sur les produits et les utilisateurs (Graph-RAG : recherche sémantique + graphe)
- **Communautés** d'utilisateurs (Louvain) et **évaluation** des modèles

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les fichiers générés par le notebook doivent se trouver dans `artifacts/`.
