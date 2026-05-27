# External Data Integration Notes

This folder records direct downloads and access attempts for Indian geoscience,
soil, and stabilized-soil data sources requested for the manuscript.

## Use in the manuscript

- Direct laboratory UCS/CBR rows from Zenodo 10.5281/zenodo.19242690 are suitable
  for model training/benchmarking after duplicate checks against the existing
  compiled dataset.
- Data.gov.in Soil Health Card organic-carbon records, if accessed through a
  registered API key or direct CSV, should be treated as regional context only.
  They are not mechanical UCS/CBR laboratory tests.
- NGDR, Bhukosh, Bhuvan, SLUSI, and CRRI are useful as context or future data
  acquisition routes, but raw mechanical stabilization data were not available as
  direct bulk downloads from static public pages during this automated audit.

## Reviewer-safe use

Do not merge portal metadata directly into UCS/CBR model rows. Use these sources
only for source-level descriptors such as region, geology, soil organic-carbon
context, slope, salinity/waterlogging risk, or future data-availability discussion
unless actual tabular laboratory measurements are downloaded.
