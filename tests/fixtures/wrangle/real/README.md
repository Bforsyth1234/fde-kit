# Real government files

Downloaded 2026-10-30 from US federal statistical agencies. Works of the US federal government
are in the public domain. These are here because the synthetic fixtures one directory up only
test the mess we thought of; these test the mess we didn't.

| File | Source | Bytes | What makes it hard |
| --- | --- | --- | --- |
| `NST-EST2025-POP.xlsx` | Census, state population totals 2020–2025: https://www2.census.gov/programs-surveys/popest/tables/2020-2025/state/totals/NST-EST2025-POP.xlsx | original | merged full-width title rows; a header row of numbers (years) under a merged label; `.Alabama` leading dots; footnote rows below the data |
| `table_5_06_a.xlsx` | EIA Electric Power Monthly table 5.6.A: https://www.eia.gov/electricity/monthly/xls/table_5_06_a.xlsx | original | two-row header (sector over month); division rows mixed with states; `--` for not applicable |
| `mrtssales92-present.xlsx` | Census Monthly Retail Trade: https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx | original | one sheet per year; NOT ADJUSTED and ADJUSTED tables on one sheet; split header (labels in row 4, months in row 5); `(S)` / `(NA)` suppressed values; `May 2024` beside `Jan. 2024` |
| `co-est2025-alldata.slice.csv` | Census county population: https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv | **slice**: the header line plus every Connecticut and New Mexico line, bytes copied unchanged | Latin-1, not UTF-8 (`Doña Ana County` is byte 0xF1); state rows (`SUMLEV 040`) mixed with county rows |
| `generation_monthly.extract.xlsx` | EIA monthly generation by state: https://www.eia.gov/electricity/data/state/generation_monthly.xlsx | **extract**: sheet `2024_Final`, the 5 rows above the data plus every CT and RI row, re-saved with openpyxl (the original is 19 MB) | 4 title rows above the header; year and month in two separate columns; `Total` rows that are not subtotals of the rows above |

`SHA256SUMS` pins every file. The 19 MB original generation workbook is not committed; the read
time and memory measured on it are in docs/FINDINGS.md.
