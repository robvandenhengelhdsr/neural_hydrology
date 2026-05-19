# HDSR Afvoervoorspelling met Neural Hydrology

Dit project gebruikt deep learning (LSTM) modellen om afvoeren te voorspellen voor de afvoergebieden van Hoogheemraadschap De Stichtse Rijn (HDSR). Het project is gebaseerd op de [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) bibliotheek.

## Overzicht

Het project bevat experimenten met verschillende LSTM varianten voor het voorspellen van afvoeren in 40 polders/afvoergebieden binnen het beheergebied van HDSR. De modellen gebruiken meteorologische data en gebiedskenmerken om accurate afvoervoorspellingen te maken.

## Projectstructuur

De **Git repo-root** bevat projectconfiguratie en data; Python-code staat onder `src/neural_hydrology/` (gangbare src-layout).

```
neural_hydrology/                    # Git repo-root
├── pyproject.toml
├── README.md
├── config.yml                       # NeuralHydrology run-config
├── requirements.txt
├── .env.example
├── data/                            # Voorbeeld-/trainingsdata
├── data_ens/                        # Operationele ensemble-input
├── runs/                            # Modelruns (lokaal, gitignored)
├── inference_runs/                  # Inference-output (gitignored)
├── notebooks/
└── src/
    └── neural_hydrology/            # Python package (pip install -e .)
        ├── paths.py                 # Pad- en .env-resolutie
        ├── preprocessing/
        ├── training/
        ├── inference/
        ├── analysis/
        ├── viz/
        └── utils/
```

## Datasets

### Afvoergebieden

Het project werkt met 40 afvoergebieden van HDSR. De lijst staat in `data/hdsr_polders.txt`.

### Data bestanden

- **Attributes**: `polders_data_aangevuld.csv` - Gebiedskenmerken van alle polders
- **Time series**: NetCDF bestanden (`.nc`) met meteorologische en hydrologische tijdreeksen per polder
- **Voorbeelden**: Alleen AFVG1, AFVG13 en AFVG15 zijn meegeleverd als voorbeelden (vanwege bestandsgrootte)
- **Let op**: De volledige `time_series/` (NetCDF) dataset wordt in Databricks gebruikt vanaf een Volume (zie `config.yml`) en zit niet in deze repository.

## Model varianten

Het project test verschillende LSTM configuraties:

1. **MTSLSTM** - Multi-Timescale LSTM
2. **MTSLSTM + Embedding** - Met embedding layer voor categorische features
3. **MTSLSTM + One-Hot Encoding** - Met one-hot encoded features
4. **Statische Multi-Timescale LSTM** - Varianten met statische features

De configuratie van een run staat in `config.yml`. Voor experimenten maak je hiervan doorgaans varianten (bijv. per model/feature-set).

## Belangrijkste scripts

Alle modules staan onder `src/neural_hydrology/`. Start ze na `pip install -e .` met `python -m neural_hydrology.<module>`.

### Preprocessing (operationeel)

- `preprocessing.create_timeseries_files` — bouwt `data_ens/time_series/<SHAPE_ID>.nc` met 30-leden ensembles

### Training

- `training.run_model` — basis training met `config.yml`
- `training.batch_train_single` — batch training per afvoergebied
- `training.hyperparameter_optimalisatie` — Optuna HPO (Databricks + MLflow)
- `training.batch_train_model` — retrain op gekozen HPO-trial

### Inference (operationeel)

- `inference.run_model` — ensemble inference naar `inference_runs/`

### Analyse

- `analysis.best_model` — evaluatie beste modellen
- `analysis.map_hdsr` — visualisatie HDSR-gebied

## Lokaal gebruik

### Installatie

```bash
cd neural_hydrology          # Git repo-root (deze map)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

Code staat in `src/neural_hydrology/`; `config.yml`, `.env` en data staan op **repo-root** (de Git folder).

### Configuratie (`.env`)

Eén bestand op repo-root: `cp .env.example .env`. Paden worden opgelost via `neural_hydrology.paths` (`src/neural_hydrology/paths.py`).

**Regel:** shell-omgevingsvariabelen overschrijven `.env`. Relatieve paden (`data`, `data_ens`, …) zijn relatief aan `NEURAL_HYDROLOGY_ROOT`.


| Variabele                 | Lokaal                                          | Databricks                                                                                     |
| ------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `NEURAL_HYDROLOGY_ROOT`   | leeg (auto repo-root)                           | `/Workspace/Shared/neural_hydrology_fork`                                                      |
| `DATA_DIR`                | `data`                                          | optioneel: `/Volumes/.../data_neuralhydrology/input`                                           |
| `DATA_ENS_DIR`            | `data_ens`                                      | idem of pad op Volume                                                                          |
| `INFERENCE_RUNS_DIR`      | `inference_runs`                                | idem of pad op Volume                                                                          |
| `BEST_MODEL_DIR`          | optioneel; leeg = nieuwste run onder `RUNS_DIR` | pad naar getrainde run (bevat `config.yml`)                                                    |
| `N_ENSEMBLES`             | `30`                                            | aantal ensembleleden bij inference                                                             |
| `ENSEMBLE_FORECAST_TABLE` | — (niet lokaal)                                 | verplicht voor postprocess-job; bv. `dbw_datascience_tst_weu_001.default.ensemble_forecast_1h` |
| `CONFIG_PATH`             | `config.yml`                                    | `/Workspace/Shared/neural_hydrology_fork/config.yml`                                           |
| `RUNS_DIR`                | `runs`                                          | idem of pad op Volume                                                                          |
| `OUTPUT_DIR`              | `runs` (default)                                | `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output`                     |
| `BASE_CONFIG`             | leeg (= `CONFIG_PATH`)                          | `/Workspace/Shared/neural_hydrology_fork/config.yml`                                           |
| `HPO_OUTPUT_DIR`          | leeg (= `OUTPUT_DIR/HPO`)                       | leeg (= `OUTPUT_DIR/HPO`)                                                                      |
| `RETRAIN_BASE_DIR`        | leeg (= `OUTPUT_DIR/BATCH_RETRAIN`)             | leeg (= `OUTPUT_DIR/BATCH_RETRAIN`)                                                            |
| `MLFLOW_TRACKING_URI`     | leeg                                            | `databricks`                                                                                   |
| `KNMI_API_URL`            | `https://api.dataplatform.knmi.nl/open-data`    | zelfde                                                                                         |
| `KNMI_API_KEY`            | verplicht (preprocessing)                       | verplicht                                                                                      |
| `ENSEMBLE_STARTTIME`      | optioneel, bv. `2026040718`                     | optioneel                                                                                      |
| `DOWNLOAD_ENSEMBLE`       | `1` (download) of `0` (cache)                   | `1` of `0`                                                                                     |
| `SOURCE_EXPERIMENT_NAME`  | optioneel (`best_model.py`)                     | optioneel, bv. `/Shared/hdsr_lstm_optuna_...`                                                  |
| `SOURCE_TRIAL_NUMBER`     | `0` (default)                                   | trial-nummer MLflow                                                                            |


Zie ook inline uitleg in `[.env.example](.env.example)`.

**Lokaal**

```bash
pip install -e .
cp .env.example .env
python -m neural_hydrology.preprocessing.create_timeseries_files --days 30 --basin-id AFVG1
python -m neural_hydrology.training.run_model
```

**Databricks**

```bash
cd /Workspace/Shared/neural_hydrology_fork
pip install -e .
python -m neural_hydrology.training.hyperparameter_optimalisatie
```

### Training

```bash
python -m neural_hydrology.training.run_model
python -m neural_hydrology.training.batch_train_single
```

### Analyse

```bash
python -m neural_hydrology.analysis.best_model
```

## Databricks

Deze branch is bedoeld om het NeuralHydrology-framework te draaien op Databricks. De trainingsscripts zijn hierop ingericht (paden onder `Workspace` en `Catalog/Volumes`, rekenkracht via `Compute`, en MLflow tracking via `Jobs & Pipelines`). In het project is gewerkt in de instantie `dbw-datascience-tst-weu-001` binnen Databricks.

### Installatie en update van repo in Databricks

#### Repo toevoegen als Git folder (eerste keer)

- **Workspace locatie kiezen**: ga naar *Workspace* en navigeer naar de plek waar je de projectfolder wilt hebben, bijv. onder `Shared` of onder `Users` → `rob.van.den.hengel@hdsr.nl`.
- **Git folder aanmaken**: rechtermuisknop op de map waarin je het project wilt plaatsen → *Create* → *Git folder*.
- **URL plakken**: plak de Git-URL van de repository bij *URL*.
- **Naam instellen**: pas indien nodig de *Name* aan zoals die in Databricks getoond wordt. Deze naam moet **uniek** zijn binnen die locatie in Databricks.
- **Aanmaken**: klik *Create Git folder* en selecteer (indien gevraagd) direct de juiste branch.
- **In dit project**: in dit project zijn folders gebruikt onder `Workspace/Shares/neural_hydrology` en `Workspace/Shares/neural_hydrology_fork`. `neural_hydrology_fork` is gebruikt vanwege beperkte rechten op het GitHub-account `hdsr-mid`, en zodoende om via een gesyncte fork te werken met het account `robvandenhengelhdsr`.

#### Repo updaten (na wijzigingen buiten Databricks)

- **Navigeer naar de Git folder**: ga in *Workspace* naar de Git folder van het project.
- **Open Git-menu**: klik op het Git-icoon rechts naast de naam van de Git folder (knop met Git-logo + branchnaam).
- **Lokale wijzigingen eerst veiligstellen (best practise)**: als er in Databricks lokale wijzigingen zijn, commit en push die eerst naar GitHub voordat je gaat pullen.
- **Branch kiezen**: selecteer linksboven de branch die je uit GitHub wilt ophalen/gaan gebruiken in Databricks.
- **Pull uitvoeren**: klik rechtsboven op *Pull* om de laatste wijzigingen op te halen.
- **Configureer `.env`**: zie [Configuratie (`.env`)](#configuratie-env) → Databricks.

#### Data Volume (input/output)

- **Dataset**: de repo bevat niet de trainingsdataset vanwege de omvang. Binnen Databricks zijn datasets beschikbaar via de *Catalog*.
- **Volume**: voor dit project is in de Catalog een Volume aangemaakt in de database `default` met de naam `data_neuralhydrology`.
  - **Input**: subfolder `input` bevat de benodigde gegevens (tijdseries & gebiedskenmerken) voor training.
  - **Output**: subfolder `output` is bedoeld voor het wegschrijven van modelresultaten.

#### Compute aanmaken of starten

- **Computeless werken**: je kunt (deels) zonder compute werken, maar dan kun je bijvoorbeeld geen terminal openen om via command line te werken en niet alle workloads draaien.
- **Nieuwe compute**: ga naar *Compute* → *Create compute* en volg de instructies. Voor GPU-training kies je een cluster met CUDA/GPU runtime. De scripts schakelen automatisch tussen GPU/CPU op basis van `torch.cuda.is_available()`.
- **Bestaande compute starten**: ga naar *Compute* en klik op het *Play*-icoon (driehoek naar rechts) bij de gewenste compute (verschijnt bij hover).

#### Libraries installeren op de compute

- Klik op de compute → tab *Libraries* → *Install new*.
- Installeer het project editable vanuit de Git folder: `pip install -e /Workspace/Shared/<jouw-git-folder>` (of via terminal in de repo-root).
- Zorg dat `.env` op repo-root het Databricks-blok bevat (zie `.env.example`).

#### Script als Job draaien (optioneel, aanbevolen voor reproduceerbare runs)

- **Nieuwe job aanmaken**:
  - Ga naar *Jobs & Pipelines* → *Create* → *Job*.
  - Kies het juiste task type: *Python script*, *Notebook* of *Add another task type*. In dit project is dit meestal *Python script*.
  - Vul *Task name* in.
  - Kies onder *Compute* de compute waarop de job moet draaien.
  - Kies onder *Path* het `.py` script dat je wilt uitvoeren en selecteer dit.
  - Klik *Create task*.
- **Job starten**:
  - Controleer of de compute aan staat.
  - Open de job en klik rechtsboven op *Run now*.

### Uitvoeren van runs voor training (incl. opzet hyperparameter optimalisatie)

#### Training van één run

- **Configuratie**: `config.yml` in de Git repo-root is de centrale config.
  - **Data**: in deze branch wijst `data_dir` naar een Databricks Volume genaamd `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/input`.
  - **Outputs**: Op Databricks schrijft de repo outputs naar een Volume genaamd `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output`.
- **Run starten** (na `pip install -e .` in de Git folder):

```bash
python -m neural_hydrology.training.run_model
```

#### Hyperparameter optimalisatie (Optuna + MLflow)

Voor HPO:

```bash
python -m neural_hydrology.training.hyperparameter_optimalisatie
```

- **MLflow**: het script zet `MLFLOW_TRACKING_URI=databricks` en logt naar een experiment onder `/Shared/...`.
- **Belangrijke instellingen in het script**:
- `BASE_CONFIG`: pad naar de basis `config.yml` die per trial wordt aangepast.
- `OUTPUT_DIR` en `RUNS_DIR`: outputlocaties op een Volume (standaard onder `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output`).
- `N_TRIALS`: aantal Optuna trials.
- **Wat er gebeurt**:
- Per trial wordt een eigen config geschreven en als MLflow artifact gelogd.
- NeuralHydrology wordt gestart via `start_run(...)`.
- De output van elke trial komt in een eigen trial-map terecht, met daarbinnen de daadwerkelijke run-folder van NeuralHydrology.
- De objective leest validatie-metrics uit TensorBoard logs, gebruikt tags zoals `valid/mean_nse_1D` en `valid/mean_nse_1h`, en optimaliseert op de maximale gemiddelde NSE over beide frequenties.

#### Batch retraining van een gekozen HPO-trial

Voor het opnieuw trainen van een specifieke HPO-trial:

```bash
python -m neural_hydrology.training.batch_train_model
```

- **MLflow**: het script zet `MLFLOW_TRACKING_URI=databricks` en logt de retrain-runs naar een apart MLflow experiment.
- **Belangrijke instellingen in het script**:
- `EXPERIMENT_NAME`: naam van de HPO-experimentmap onder `.../output/HPO/`.
- `TRIAL_NAME`: de trial-map die je wilt hertrainen, bijvoorbeeld `trial_28`.
- `PATH_HPO`: pad naar de HPO-output waarin de gekozen trial staat.
- `RETRAIN_NAME`: naam/suffix voor de retrain-run(s).
- `NUMBER_OF_RETRAININGS`: aantal keer dat dezelfde trial opnieuw wordt getraind.
- `RETRAIN_BASE_DIR` en `DESTINATION_DIR`: outputlocaties voor de gekopieerde run en de nieuwe retrains.
- **Wat er gebeurt**:
- Het script zoekt eerst de gekozen trial op in de HPO-output en bepaalt de bijbehorende NeuralHydrology run-folder.
- Die run-folder wordt gekopieerd naar een aparte retrain-locatie.
- Vervolgens wordt per retraining een nieuwe config geschreven met een nieuw `experiment_name`, maar op basis van de gekozen HPO-trial.
- NeuralHydrology wordt opnieuw gestart via `start_run(...)`.
- Na iedere retrain worden de validatie-metrics uit TensorBoard gelezen en in MLflow gelogd, zodat meerdere retrains van dezelfde trial onderling vergeleken kunnen worden.

### Operationeel verwachtingen

Deze sectie beschrijft de **operationele keten** op Databricks: (1) ensemble-tijdreeksen opbouwen, (2) ensemble-inference met een getraind model, (3) resultaten naar Unity Catalog.

#### 1) Preprocessing: ensemble tijdreeksen bouwen

Het module `neural_hydrology.preprocessing.create_timeseries_files` bouwt per afvoergebied (`SHAPE_ID`) één NetCDF in `data_ens/time_series/<SHAPE_ID>.nc` met **30 ensembleleden** per variabele (`neerslag_1` … `neerslag_30`, idem voor `temperatuur`, `u`, `v`, `straling`, `streefpeil`).

- **Streefpeil** — constant over de volledige `date`-as: laatste niet-NaN waarde uit `DATA_DIR/time_series/<SHAPE_ID>.nc` (training), identiek in `streefpeil_1` … `streefpeil_30` (`units`: `mNAP`).

##### Bronnen en volgorde

1. **Historisch meteo (Cabauw)** — KNMI **klimatologie uurgegevens** (station 348 Cabauw) + waar nodig aanvulling uit KNMI Open Data **10-minuut** stationdata: temperatuur, zonnestraling, wind als `u`/`v`.
2. **Historisch neerslag** — KNMI **MFBS** (radar, uur) gecombineerd met **RTCOR** (5-min → **uursom (mm)** per gebied; aggregatie met minimum-aantal 5-min stappen).
3. **Forecast** — KNMI Open Data **HARMONIE CY43**: composiet uit **twee datasets** (`harmonie_arome_cy43_p2a` meteo + `harmonie_arome_cy43_p2b` renew/straling), tot **30 leden** over een rollend 6-uurs venster (`ENSEMBLE_STARTTIME` in `.env` op repo-root).

##### KNMI in `.env`

`KNMI_API_KEY` (verplicht), optioneel `ENSEMBLE_STARTTIME` (`YYYYMMDDHH` UTC), `DOWNLOAD_ENSEMBLE` (`1` = download, `0` = alleen cache in `data_ens/_tmp_harmonie/`).

##### Gedrag bij ontbrekende waarden (na merge historisch + forecast)

- **Temperatuur, straling, wind (`u`,`v`)**: korte **lineaire interpolatie** langs de tijd; maximum gap wordt bepaald door `**METEO_INTERP_LIMIT_HOURS`** in `create_timeseries_files.py`.
- **Neerslag**: resterende **NaN → 0** (neerslag is een **uursom per tijdstap**, dus in de praktijk **mm per uurstap**; in de NetCDF staat het `units`-attribuut momenteel als `"mm"`). Dit gedrag is aan/uit via `MissingDataConfig.neerslag_fill_nan_with_zero`.

##### Overige instellingen in het script

- `**RTCOR_MAX_DOWNLOADS`** — maximum aantal RTCOR-bestandsdownloads per run (bescherming tegen te lange KNMI-pulls).

##### Uitvoeren

```bash
# Na pip install -e . in de repo-root:
python -m neural_hydrology.preprocessing.create_timeseries_files --days 365
python -m neural_hydrology.preprocessing.create_timeseries_files --days 30 --basin-id AFVG1
```

**Databricks Job:** `python jobs/preprocess_ensembles.py`

```mermaid
flowchart TD
  subgraph knmi_hist [KNMI historisch]
    knmi_uur[KNMI klimatologie uurgegevens Cabauw STN348]
    knmi_10m[KNMI Open Data 10-min station waar nodig]
    knmi_mfbs[KNMI Open Data MFBS radar uur]
    knmi_rtcor[KNMI Open Data RTCOR 5-min]
  end

  subgraph knmi_fc [KNMI forecast]
    knmi_p2a[KNMI Open Data HARMONIE CY43 p2a meteo]
    knmi_p2b[KNMI Open Data HARMONIE CY43 p2b renew]
  end

  knmi_uur --> cabauw[Cabauw loader]
  knmi_10m --> cabauw
  cabauw --> histMeteo[Hist meteo DataFrame]

  knmi_mfbs --> radar[Radar MFBS plus RTCOR loader]
  knmi_rtcor --> radar
  radar --> histPrec[Hist neerslag per gebied]

  knmi_p2a --> harmonie[HARMONIE ensemble compositie]
  knmi_p2b --> harmonie
  harmonie --> fcEns[Forecast arrays tijd x 30 leden]

  histMeteo --> orchestrator["create_timeseries_files.py"]
  histPrec --> orchestrator
  fcEns --> orchestrator
  orchestrator --> basinNc["data_ens/time_series SHAPE_ID.nc"]
```



#### 2) Inference: ensemble verwachtingen draaien met getraind model

`neural_hydrology.inference.run_model` draait **ensemble inference** met een eerder getrainde NeuralHydrology run (map met `config.yml` en checkpoints) op de NetCDF’s uit `data_ens/time_series/`.

- **Input**
  - **Modelrun**: `BEST_MODEL_DIR` in `.env`, of `--run_dir <pad/naar/runs/<run_id>>` (moet `config.yml` bevatten)
  - **Data**: `--data_dir` default = `data_ens/` op repo-root (via `DATA_ENS_DIR` in `.env`)
  - **Basin lijst**: optioneel `--basin_file <pad>` (default: `<data_dir>/hdsr_polders.txt`)
  - **Ensemble starttijd**: `ENSEMBLE_STARTTIME=YYYYMMDDHH` (UTC) in `.env` (optioneel)
  - **Aantal leden**: `N_ENSEMBLES` in `.env` (default `30`)
- **Uitvoering**
  - Bepaalt automatisch een **testperiode** op basis van de NetCDF-periode en de benodigde **warm-up** uit de trainingconfig (`seq_length` en `predict_last_n`, voor alle `use_frequencies`).
  - Loopt per ensemblelid k = 1..N en selecteert inputs via kolommen `<variabele>_<k>` uit de NetCDF (bijv. `neerslag_17`), die intern als `neerslag` etc. worden aangeboden aan het model.
- **Output**
  - Schrijft per frequentie één NetCDF in `inference_runs/`:
    - `inference_runs/polders_hdsr_<freq>.nc`
  - De NetCDF bevat **groepen per basin** (groepnaam = `SHAPE_ID`) met een `datetime`-as en variabelen per ensemblelid:
    - `<target>_sim_1`, `<target>_sim_2`, … `<target>_sim_<N>`

##### Uitvoeren

```bash
# Voorbeeld: 30-leden ensemble inference (vanuit repo-root, na pip install -e .)
python -m neural_hydrology.inference.run_model \
  --run_dir runs/<jouw_run_map> \
  --data_dir data_ens \
  --out_dir inference_runs \
  --n_ensembles 30
```

**Databricks Job** (paden uit `.env`, na `pip install -e .`):

```bash
python jobs/run_ensembles.py
```

#### 3) Postprocess: ensemble-resultaat naar Unity Catalog

`jobs/postprocess_ensembles.py` leest `INFERENCE_RUNS_DIR/polders_hdsr_1h.nc` en schrijft alle basins en ensembleleden naar een Delta-tabel in Unity Catalog.

- **Input**: `polders_hdsr_1h.nc` (vast) onder `INFERENCE_RUNS_DIR`
- **Config**: `ENSEMBLE_FORECAST_TABLE` in `.env` (verplicht op Databricks), drie-delige naam `catalog.schema.table`
- **Refresh**: bij elke run eerst `TRUNCATE TABLE`, daarna volledige reload uit de NetCDF-periode
- **Kolommen**: `datetime`, `ensemble_id`, `afvoergeb_id`, `value`
- **Power BI**: rapport en handleiding in [`../powerbi/`](../powerbi/) (tabel `dbw_datascience_tst_weu_001.default.output_forecast`)

```bash
python jobs/postprocess_ensembles.py
```

**Volledige keten (Databricks Jobs):**

```bash
python jobs/preprocess_ensembles.py
python jobs/run_ensembles.py
python jobs/postprocess_ensembles.py
```

## Configuratie

De run-config staat in `config.yml`. Deze definieert o.a.:

- Model architectuur (LSTM variant)
- Input features
- Training parameters
- Data preprocessing
- Output metrics

## Resultaten

De training resultaten worden lokaal opgeslagen in een `runs/` folder (niet meegeleverd vanwege grootte). Elke run bevat:

- Getrainde model checkpoints
- Evaluatie metrics
- Visualisaties
- TensorBoard logs

## Notebooks

- `hyperparameter_importance.ipynb` - Analyse van hyperparameter importance en model performance

## Licentie

Dit project is ontwikkeld voor onderzoek binnen HDSR. Voor gebruik van de neuralhydrology bibliotheek, zie de [originele licentie](https://github.com/neuralhydrology/neuralhydrology/blob/master/LICENSE).

## Referenties

- Kratzert, F., et al. (2019). "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets." Hydrology and Earth System Sciences.
- [NeuralHydrology Documentatie](https://neuralhydrology.readthedocs.io/)
- [KNMI HARMONI Documentatie](https://www.knmidata.nl/open-data/harmonie)

