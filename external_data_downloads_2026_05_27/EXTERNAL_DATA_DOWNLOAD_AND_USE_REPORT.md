# External Data Download and Use Report

## Summary

The external-source audit separated direct machine-learning data from contextual
or manually gated sources. Only files containing actual UCS/CBR laboratory rows
were placed in `processed/` for possible modelling. Portal pages, service
descriptions, or blocked sources were kept in `metadata/` and documented in the
blocker table.

## Directly usable data

Zenodo record `10.5281/zenodo.19242690` was successfully downloaded through the
Zenodo API. The workbook was converted into separate CSV files:

sheet,rows,columns,csv
CBR_clean,163,59,D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML\External_Data_Downloads_2026-05-27\processed\zenodo_19242690_CBR_clean.csv
UCS_clean,397,61,D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML\External_Data_Downloads_2026-05-27\processed\zenodo_19242690_UCS_clean.csv
StudyInventory,20,14,D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML\External_Data_Downloads_2026-05-27\processed\zenodo_19242690_StudyInventory.csv
DataDictionary,72,8,D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML\External_Data_Downloads_2026-05-27\processed\zenodo_19242690_DataDictionary.csv
VocabularyMap,23,6,D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML\External_Data_Downloads_2026-05-27\processed\zenodo_19242690_VocabularyMap.csv
QC_Log,62,10,D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML\External_Data_Downloads_2026-05-27\processed\zenodo_19242690_QC_Log.csv


## Data.gov.in status

The Soil Health Card organic-carbon resource metadata page was downloaded. The
page exposes a file reference and API route, but the automated API route returned
`Meta not found`, and the direct CSV route returned an HTML page rather than
tabular CSV during this run. This source should therefore be treated as metadata
or manually downloadable context until a registered Data.gov.in API key or a
working direct file download is available.

## NGDR, Bhukosh, Bhuvan, SLUSI, and CRRI status

These sources are important for future enrichment, especially geology,
geochemistry, soil survey, slope, waterlogging, and regional soil-condition
layers. However, raw mechanical stabilization test data were not exposed as
static public downloads during this automated audit. They should not be merged
into the UCS/CBR training table unless actual tabular laboratory measurements
are obtained.

## Reviewer-safe recommendation

Use Zenodo rows as direct experimental additions only after duplicate checking
against the existing compiled dataset. Use the Indian government portals as
contextual data sources or as future data-acquisition pathways. This distinction
prevents overclaiming and protects the manuscript from criticism that metadata
or regional context was treated as laboratory strength data.
