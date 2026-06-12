| Profile              | Policy                           | Mobility success | HO attempt success | Outage ratio | SINR p5 (dB) | HO/km  | Ping-pong | Overlap interruption |
| -------------------- | -------------------------------- | ---------------: | -----------------: | -----------: | -----------: | -----: | --------: | -------------------: |
| distance_2km         | Fixed A3 (Hys=3, TTT=150)        | 0.885            | 0.906              | 0.023        | -1.765       | 1.950  | 0.350     | 0.017                |
| distance_2km         | Position-prior A3                | 0.892            | 0.910              | 0.019        | -1.853       | 1.300  | 0.150     | 0.011                |
| distance_2km         | Speed-adaptive A3                | 0.851            | 0.863              | 0.027        | -1.545       | 2.200  | 0.500     | 0.027                |
| distance_2km         | Aggressive A3 (Hys=2.5, TTT=100) | 0.855            | 0.860              | 0.029        | -1.823       | 2.300  | 0.500     | 0.027                |
| distance_2km         | Rainbow-GRU + late guard         | 0.814            | 0.950              | 0.021        | -2.354       | 1.050  | 0.000     | 0.001                |
| distance_2km         | Tabular Q-learning               | 0.740            | 0.759              | 0.038        | -2.905       | 3.000  | 1.000     | 0.035                |
| distance_2km         | Signal-trend guard A3            | 0.701            | 0.701              | 0.039        | -3.237       | 3.100  | 0.900     | 0.032                |
| distance_2km         | Oracle A3 grid                   | 0.733            | 1.000              | 0.024        | -2.463       | 0.550  | 0.000     | 0.000                |
| distance_2km         | Rainbow-GRU                      | 0.603            | 1.000              | 0.036        | -3.506       | 0.800  | 0.000     | 0.001                |
| distance_4km         | Fixed A3 (Hys=3, TTT=150)        | 0.940            | 0.959              | 0.017        | -1.235       | 1.500  | 0.100     | 0.010                |
| distance_4km         | Position-prior A3                | 0.878            | 0.940              | 0.017        | -1.648       | 1.050  | 0.100     | 0.003                |
| distance_4km         | Speed-adaptive A3                | 0.828            | 0.854              | 0.024        | -1.237       | 1.925  | 0.300     | 0.014                |
| distance_4km         | Aggressive A3 (Hys=2.5, TTT=100) | 0.806            | 0.815              | 0.028        | -1.540       | 2.200  | 0.600     | 0.019                |
| distance_4km         | Rainbow-GRU + late guard         | 0.827            | 1.000              | 0.018        | -2.469       | 0.875  | 0.000     | 0.005                |
| distance_4km         | Tabular Q-learning               | 0.772            | 0.797              | 0.031        | -2.475       | 2.525  | 1.200     | 0.025                |
| distance_4km         | Signal-trend guard A3            | 0.658            | 0.660              | 0.033        | -2.227       | 2.675  | 0.950     | 0.024                |
| distance_4km         | Oracle A3 grid                   | 0.609            | 1.000              | 0.021        | -2.634       | 0.375  | 0.000     | 0.000                |
| distance_4km         | Rainbow-GRU                      | 0.578            | 1.000              | 0.026        | -3.372       | 0.625  | 0.000     | 0.005                |
| speed_400            | Fixed A3 (Hys=3, TTT=150)        | 0.870            | 0.896              | 0.025        | -2.099       | 1.567  | 0.250     | 0.025                |
| speed_400            | Position-prior A3                | 0.851            | 0.909              | 0.021        | -2.259       | 0.933  | 0.150     | 0.011                |
| speed_400            | Speed-adaptive A3                | 0.778            | 0.778              | 0.036        | -2.975       | 2.200  | 0.550     | 0.035                |
| speed_400            | Aggressive A3 (Hys=2.5, TTT=100) | 0.837            | 0.854              | 0.030        | -2.188       | 1.867  | 0.500     | 0.034                |
| speed_400            | Rainbow-GRU + late guard         | 0.794            | 0.980              | 0.029        | -3.422       | 0.867  | 0.000     | 0.002                |
| speed_400            | Tabular Q-learning               | 0.798            | 0.817              | 0.031        | -2.352       | 1.967  | 0.650     | 0.036                |
| speed_400            | Signal-trend guard A3            | 0.693            | 0.693              | 0.036        | -3.316       | 2.233  | 0.550     | 0.028                |
| speed_400            | Oracle A3 grid                   | 0.785            | 1.000              | 0.015        | -2.037       | 0.367  | 0.000     | 0.001                |
| speed_400            | Rainbow-GRU                      | 0.411            | 0.983              | 0.040        | -4.782       | 0.400  | 0.000     | 0.000                |
| stress_radio_holdout | Fixed A3 (Hys=3, TTT=150)        | 0.518            | 0.631              | 0.144        | -10.857      | 2.800  | 0.250     | 0.004                |
| stress_radio_holdout | Position-prior A3                | 0.298            | 0.536              | 0.165        | -11.993      | 2.867  | 0.400     | 0.004                |
| stress_radio_holdout | Speed-adaptive A3                | 0.461            | 0.520              | 0.144        | -12.723      | 5.400  | 0.750     | 0.015                |
| stress_radio_holdout | Aggressive A3 (Hys=2.5, TTT=100) | 0.402            | 0.460              | 0.156        | -13.711      | 6.200  | 0.900     | 0.019                |
| stress_radio_holdout | Rainbow-GRU + late guard         | 0.418            | 0.807              | 0.158        | -11.431      | 1.633  | 0.000     | 0.000                |
| stress_radio_holdout | Tabular Q-learning               | 0.481            | 0.583              | 0.147        | -11.572      | 3.833  | 0.650     | 0.011                |
| stress_radio_holdout | Signal-trend guard A3            | 0.311            | 0.315              | 0.207        | -19.642      | 18.733 | 5.700     | 0.092                |
| stress_radio_holdout | Oracle A3 grid                   | 0.220            | 0.834              | 0.174        | -11.873      | 0.767  | 0.000     | 0.000                |
| stress_radio_holdout | Rainbow-GRU                      | 0.142            | 0.850              | 0.200        | -13.717      | 0.433  | 0.000     | 0.000                |
