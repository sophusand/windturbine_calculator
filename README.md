# Vindmølle Beregner (Streamlit)

En interaktiv vindmølle-beregner med grafer, eksport og delbare indstillinger.

## Kom i gang

### macOS / Linux
1. Åbn en terminal i projektmappen
2. Opret et virtuelt miljø og installer pakker
   - `python3 -m venv venv`
   - `source venv/bin/activate`
   - `pip install -r requirements.txt`
3. Start appen
   - `streamlit run windturbine_calculator.py`

### Windows (PowerShell)
1. Åbn PowerShell i projektmappen
2. Opret et virtuelt miljø og installer pakker
   - `py -m venv venv`
   - `venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`
3. Start appen
   - `streamlit run windturbine_calculator.py`

## Del indstillinger
Du kan downloade dine indstillinger som JSON i sidebaren og sende filen til andre. De kan importere filen direkte i appen for at få samme konfiguration.

## Fejlfinding
- Hvis `streamlit` ikke findes, så kør `pip install -r requirements.txt` igen.
- Sørg for at bruge Python 3.9+.
