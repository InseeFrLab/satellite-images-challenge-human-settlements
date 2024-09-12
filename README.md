# Human Settlements Challenge

Challenge mexicain visant à détecter automatiquement des installations humaines sur des images satellites du Mexique :  [Lien vers le challenge sur Zindi](https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge)

```bash
git clone https://github.com/InseeFrLab/satellite-images-challenge-human-settlements.git
cd satellite-images-challenge-human-settlements/
pip install -r requirements.txt
```

Le notebook start_challenge.ipynb permet de prendre en main les données.

## Lancer un entraînement

Ouvrir un service vs-code-pytorch avec ou sans gpu en ayant accès au projet slums detection sur onyxia.
Modifier le fichier **config.yml** pour l'adapter à l'entraînement souhaité en jouant sur les données en entrée et les hyperparamètres.
Lancer l'entraînement :
```bash
cd src/
python run.py "nom_du_run_mlflow_choisi"
```