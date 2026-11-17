# Data Quality Report

| Step | Rows affected |
|---|---:|
| Raw rows | 9,120 |
| Exact duplicate rows removed | 120 |
| Duplicate ticket_id rows removed | 0 |
| Rows with empty/junk ticket text removed | 30 |
| Category labels standardised (case/spaces/underscores) | 559 |
| Channel labels standardised | 343 |
| Unparseable created_at dates | 0 |
| Invalid first_response_min values (<=0 or >7 days) set to null | 25 |
| first_response_min imputed with channel median | 293 |
| Missing city filled with 'Unknown' | 357 |
| Missing customer_tier filled with 'Unknown' | 179 |
| Clean rows | 8,970 |
