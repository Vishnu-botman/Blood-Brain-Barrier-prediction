This folder is optional. The uploaded BBBP raw CSV contained **no organizer split**; the default command produces and records one seed-42 scaffold split from `data_sources/bbbp.csv`.

If the teacher supplies official fixed splits later, put their unchanged `train.csv`, `valid.csv`, `test.csv` here and use `python run_hackathon.py --official-data-dir official_data --out official_run`. Do not call the project's self-generated split official.
