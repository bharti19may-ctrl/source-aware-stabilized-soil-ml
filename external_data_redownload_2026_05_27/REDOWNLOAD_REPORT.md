# External Data Redownload Report

A second download pass was completed with network access enabled. The process separated direct laboratory datasets from contextual geospatial/agricultural records and metadata-only portal pages.

## Processed usable/contextual files

- `datagov_soil_health_oc_all_offsets.csv` (7605 bytes)
- `datagov_soil_health_offset_download_log.csv` (96 bytes)
- `zenodo_19242690_CBR_clean.csv` (53627 bytes)
- `zenodo_19242690_DataDictionary.csv` (10302 bytes)
- `zenodo_19242690_QC_Log.csv` (9933 bytes)
- `zenodo_19242690_sheet_summary.csv` (1090 bytes)
- `zenodo_19242690_StudyInventory.csv` (3356 bytes)
- `zenodo_19242690_UCS_clean.csv` (115309 bytes)
- `zenodo_19242690_VocabularyMap.csv` (1109 bytes)

## Main outcomes

- Data.gov.in Soil Health Card records were successfully downloaded through the UUID API using offsets 0, 10, 20, and 30, giving 35 State/UT records.
- Zenodo returned HTTP 403 during the second pass, but the same Zenodo workbook had already been downloaded successfully in the earlier same-day audit and has been copied into this redownload package with provenance retained.
- NGDR and the Bhukosh public GIS endpoint timed out during automated access; their use should remain manual or metadata-only until raw layer exports are obtained.
- Bhuvan, SLUSI, CRRI, and GSI Bhukosh catalog metadata were downloaded, but these are not mechanical UCS/CBR laboratory datasets.

## Reviewer-safe use

Only Zenodo UCS/CBR laboratory rows should be considered for direct modelling, and only after duplicate checks. Data.gov.in Soil Health Card records may be used as regional context for organic carbon and nutrient status. Portal metadata should be used only for data-availability discussion or future enrichment, not as mechanical strength observations.
