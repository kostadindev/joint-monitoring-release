# Third-party code

The customer-service testbed runs MESA's released benchmark (Li et al., 2026,
https://anonymous.4open.science/r/mesa/). MESA's artifact has no licence file, so its code
and data are **not** redistributed here.

To re-run the customer-service episodes, download the artifact from that page and unpack
it into `vendor/mesa/` (so that `vendor/mesa/src/` exists). Nothing else in this repository
needs it: every recorded episode and monitor call is already stored under `results/`, and
the MESA bridge tests skip when the code is absent.
