# Human Settlements Challenge

Challenge mexicain visant à détecter automatiquement des installations humaines sur des images satellites du Mexique :  [Lien vers le challenge sur Zindi](https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge)

```bash
git clone https://github.com/InseeFrLab/satellite-images-challenge-human-settlements.git
cd satellite-images-challenge-human-settlements/
pip install -r requirements.txt
```

Le notebook *notebooks/start_challenge.ipynb* permet de prendre en main les données.

## Lancer un entraînement

Ouvrir un service vs-code-pytorch avec ou sans gpu en ayant accès au projet slums detection sur onyxia.
Modifier le fichier **src/config.yml** pour l'adapter à l'entraînement souhaité en jouant sur les données en entrée et les hyperparamètres.
Lancer l'entraînement :
```bash
cd src/
python run.py "nom_du_run_mlflow_choisi"
```

Configurations possibles :  

    - bands: List[int] *bandes à conserver pour l'entraînement, entre 0 et 5 compris (par défaut, toutes les bandes = [0, 1, 2, 3, 4, 5])*  
    - augmentation: boolean *data augmentation*  
    - prop zeros: *entre 0 et 1, par défaut 0.5*  
    - len data limit: int *entre 0 et 1.1 million, par défaut None (pas de limite)*    

    - loss: *au choix dans ['crossentropy', 'bce', 'bcelogits']* 
    - optim: *au choix dans ["adam", "sgd"]*  
    - lr: *learning rate, par défaut 0.01*  
    - momentum: *par défaut 0.9*  
    - module: *au choix dans ["mobilenet_v2", "mobilenet_v3_small", "resnet18", "resnet34", "vgg11", "vgg11_bn"]*  
    - batch size: *par défaut 128*  
    - batch size test: *par défaut 128*  
    - max epochs: *par défaut 100*    

    - train prop: *par défaut 0.6*   
    - val prop: *par défaut 0.2*  
    - test prop: *par défaut 0.2*  
    - accumulate batch : *par défaut 3*  

    - samplesubmission: boolean *générer le fichier à soumettre pour le challenge, par défaut True*  
    - mlflow: boolean *utiliser mlflow, par défaut True*  


## Afficher le diaporama résumant ce challenge

[Site du diaporama](https://inseefrlab.github.io/satellite-images-challenge-human-settlements/)

