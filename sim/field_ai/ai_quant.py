"""ai_quant: an EV / pot-odds / fold-equity bot for No Limit (8-seat NLHE sit-and-go, 1 ms per decision).

Quant-dev layout:
  * import time: parse an embedded preflop table (169 hand classes x equity vs 1..7 random hands, and HU equity vs
    top-5%..100% ranges; computed offline by Monte Carlo, 2,500 deals per cell) and build the ICM function once.
  * act(): preflop is table lookups + a closed-form EV (all-ins scored in ICM points); postflop runs a
    time-boxed Monte Carlo (wall-clock deadline ~350 us, results cached per board so repeat decisions on a street
    reuse samples), then pot odds against a bet-size-adjusted equity, and a bet-vs-check EV with modelled fold
    equity.
  * observe(): a light opponent model (VPIP, PFR, fold-to-postflop-bet per seat, Bayesian priors) and a small
    cache warm-up of the postflop equity when the board is dealt.
Everything in act() is wrapped: any exception or illegal result falls back to check/fold.
"""
import random
import time

try:
    from sim.kit.agent import Agent, validate_action
    from sim.kit.cards import CARD_INT, evaluate
    from sim.kit.icm import make_icm
except ImportError:  # flat layout
    from agent import Agent, validate_action
    from cards import CARD_INT, evaluate
    from icm import make_icm

_perf = time.perf_counter_ns

# ------------------------------------------------------------------ precomputed tables (offline MC, see docstring)
_RANGES = (0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0)
_TABLE_TEXT = """AA 0.0045 0.8652 0.7213 0.6444 0.5689 0.4781 0.4425 0.3896 0.8278 0.8368 0.8508 0.8658 0.8664 0.8492 0.8480;AKs 0.0468 0.6704 0.5193 0.4084 0.3511 0.2937 0.2786 0.2588 0.4548 0.5670 0.6204 0.6530 0.6826 0.6808 0.6750;AKo 0.0437 0.6758 0.4867 0.3945 0.3126 0.2946 0.2411 0.2212 0.4402 0.5472 0.5928 0.6294 0.6532 0.6514 0.6564;
AQs 0.0347 0.6782 0.4871 0.3908 0.3406 0.3027 0.2644 0.2333 0.3964 0.4996 0.5684 0.6268 0.6514 0.6592 0.6548;AQo 0.0754 0.6338 0.4606 0.3620 0.3028 0.2497 0.2238 0.2044 0.3626 0.4920 0.5306 0.6152 0.6104 0.6376 0.6446;AJs 0.0543 0.6510 0.4994 0.3683 0.3269 0.2826 0.2436 0.2195 0.3768 0.4560 0.5092 0.5896 0.6268 0.6452 0.6706;
AJo 0.0664 0.6448 0.4583 0.3498 0.3014 0.2535 0.2008 0.1804 0.3042 0.4290 0.4920 0.5898 0.6000 0.6066 0.6400;ATs 0.0573 0.6456 0.4819 0.3671 0.3117 0.2532 0.2327 0.2101 0.3552 0.4154 0.4688 0.5552 0.6002 0.6138 0.6296;ATo 0.1146 0.6096 0.4485 0.3533 0.2554 0.2225 0.2012 0.1713 0.3062 0.3750 0.4434 0.5430 0.5856 0.6050 0.6308;
A9s 0.0950 0.6198 0.4339 0.3321 0.2786 0.2409 0.2160 0.1989 0.3050 0.3840 0.4174 0.5146 0.5770 0.6110 0.6210;A9o 0.1388 0.6042 0.4276 0.3078 0.2629 0.1964 0.1862 0.1419 0.2728 0.3564 0.3930 0.5128 0.5630 0.5832 0.5874;A8s 0.1056 0.6130 0.4406 0.3490 0.2501 0.2220 0.2230 0.1946 0.3084 0.3848 0.4038 0.5074 0.5300 0.5906 0.6190;
A8o 0.2262 0.5814 0.3907 0.3066 0.2343 0.1900 0.1633 0.1427 0.2728 0.3294 0.3676 0.4610 0.5306 0.5496 0.6108;A7s 0.0814 0.6248 0.4375 0.3256 0.2600 0.2283 0.2041 0.1759 0.3082 0.3538 0.4002 0.4938 0.5266 0.5676 0.6134;A7o 0.1599 0.5952 0.4022 0.2822 0.2151 0.1857 0.1606 0.1257 0.2818 0.3174 0.3480 0.4410 0.5366 0.5440 0.5704;
A6s 0.0784 0.6318 0.4241 0.3089 0.2520 0.1995 0.2031 0.1780 0.2952 0.3510 0.3936 0.4540 0.5210 0.5472 0.5906;A6o 0.2353 0.5792 0.3891 0.2663 0.2187 0.1699 0.1429 0.1209 0.2654 0.3144 0.3510 0.4280 0.5118 0.5266 0.5796;A5s 0.1297 0.6052 0.4177 0.3273 0.2611 0.2218 0.1865 0.1951 0.3154 0.3446 0.3840 0.4638 0.5268 0.5502 0.6198;
A5o 0.2021 0.5858 0.3996 0.2668 0.2065 0.1754 0.1480 0.1323 0.2854 0.3112 0.3526 0.4276 0.4898 0.5342 0.5666;A4s 0.1900 0.5894 0.3825 0.3051 0.2561 0.2195 0.1917 0.1622 0.3112 0.3448 0.3854 0.4444 0.5102 0.5406 0.5880;A4o 0.2624 0.5636 0.3797 0.2684 0.2216 0.1654 0.1291 0.1222 0.2674 0.3160 0.3474 0.4238 0.4818 0.5306 0.5472;
A3s 0.2081 0.5828 0.3865 0.3088 0.2583 0.2116 0.1909 0.1670 0.3164 0.3308 0.3846 0.4418 0.5000 0.5382 0.5870;A3o 0.3032 0.5538 0.3480 0.2599 0.2023 0.1721 0.1378 0.1170 0.2642 0.3144 0.3422 0.4068 0.4778 0.5088 0.5508;A2s 0.2534 0.5700 0.3939 0.2893 0.2516 0.2167 0.1859 0.1704 0.3128 0.3444 0.3712 0.4396 0.5050 0.5336 0.5750;
A2o 0.3213 0.5504 0.3663 0.2498 0.1949 0.1679 0.1249 0.1095 0.2744 0.3002 0.3574 0.4010 0.4660 0.4926 0.5548;KK 0.0090 0.8236 0.6741 0.5821 0.5131 0.4373 0.3873 0.3176 0.7128 0.7150 0.7306 0.7502 0.7810 0.7932 0.8184;KQs 0.0920 0.6228 0.4664 0.3906 0.3365 0.2817 0.2428 0.2329 0.3646 0.4038 0.4376 0.5132 0.5544 0.6002 0.6478;
KQo 0.1237 0.6070 0.4556 0.3427 0.2936 0.2613 0.2252 0.1845 0.3242 0.3442 0.3946 0.4634 0.5228 0.5792 0.6160;KJs 0.0890 0.6242 0.4682 0.3693 0.3192 0.2746 0.2450 0.2266 0.3402 0.3684 0.3984 0.4820 0.5296 0.5924 0.6218;KJo 0.1508 0.6018 0.4395 0.3425 0.2908 0.2359 0.2088 0.1855 0.2980 0.3438 0.3688 0.4372 0.5068 0.5636 0.6084;
KTs 0.0980 0.6142 0.4480 0.3582 0.2980 0.2698 0.2224 0.2167 0.3338 0.3764 0.3926 0.4792 0.5018 0.5648 0.6084;KTo 0.1780 0.5938 0.4191 0.3380 0.2571 0.2019 0.1962 0.1784 0.2890 0.3414 0.3608 0.4326 0.4804 0.5400 0.6000;K9s 0.2051 0.5846 0.4309 0.3273 0.2585 0.2212 0.2031 0.1757 0.2992 0.3404 0.3752 0.4188 0.4852 0.5490 0.5872;
K9o 0.1870 0.5902 0.3961 0.2794 0.2293 0.2003 0.1604 0.1477 0.2746 0.3238 0.3422 0.3728 0.4482 0.5060 0.5748;K8s 0.1931 0.5878 0.4162 0.3178 0.2434 0.2176 0.1965 0.1648 0.2920 0.3276 0.3684 0.4044 0.4656 0.5212 0.5870;K8o 0.2715 0.5634 0.3651 0.2722 0.2087 0.1674 0.1535 0.1162 0.2364 0.3070 0.3290 0.3752 0.4176 0.4930 0.5764;
K7s 0.2383 0.5770 0.3909 0.3006 0.2389 0.2070 0.1760 0.1704 0.2780 0.3320 0.3612 0.3884 0.4558 0.5032 0.5576;K7o 0.3454 0.5442 0.3577 0.2626 0.2131 0.1481 0.1358 0.1168 0.2302 0.2954 0.3122 0.3680 0.4116 0.4766 0.5566;K6s 0.2504 0.5702 0.3860 0.2928 0.2519 0.2119 0.1716 0.1572 0.2804 0.3466 0.3642 0.3818 0.4418 0.5098 0.5558;
K6o 0.2911 0.5574 0.3367 0.2554 0.1905 0.1471 0.1222 0.1227 0.2608 0.2944 0.3296 0.3428 0.4080 0.4808 0.5380;K5s 0.2775 0.5628 0.3697 0.2745 0.2262 0.1862 0.1740 0.1547 0.2816 0.3348 0.3428 0.3838 0.4276 0.4954 0.5422;K5o 0.3861 0.5304 0.3537 0.2360 0.1766 0.1548 0.1426 0.1144 0.2518 0.2652 0.3100 0.3618 0.4068 0.4546 0.5150;
K4s 0.3243 0.5478 0.3596 0.2890 0.2246 0.1917 0.1836 0.1487 0.2776 0.3038 0.3294 0.3798 0.4218 0.5036 0.5604;K4o 0.3771 0.5350 0.3368 0.2344 0.1803 0.1484 0.1339 0.1017 0.2470 0.2706 0.2986 0.3544 0.4012 0.4686 0.5276;K3s 0.2941 0.5572 0.3668 0.2733 0.2117 0.1961 0.1626 0.1413 0.2834 0.3292 0.3580 0.3648 0.4118 0.4574 0.5540;
K3o 0.4344 0.5128 0.3074 0.2312 0.1815 0.1399 0.1235 0.1069 0.2402 0.2914 0.2998 0.3482 0.3940 0.4466 0.4982;K2s 0.3560 0.5386 0.3393 0.2487 0.1973 0.1753 0.1577 0.1311 0.2800 0.3222 0.3376 0.3764 0.3944 0.4610 0.5368;K2o 0.4721 0.5044 0.3071 0.2340 0.1773 0.1495 0.1154 0.0992 0.2406 0.2778 0.2936 0.3280 0.3766 0.4360 0.4934;
QQ 0.0136 0.8004 0.6429 0.5410 0.4556 0.3838 0.3383 0.2952 0.5896 0.6618 0.6696 0.6994 0.7392 0.7642 0.8016;QJs 0.1267 0.6056 0.4355 0.3558 0.3053 0.2701 0.2330 0.2129 0.3548 0.3588 0.4144 0.4210 0.4748 0.5450 0.5992;QJo 0.1689 0.5948 0.4020 0.3521 0.2753 0.2124 0.2043 0.1701 0.3162 0.3510 0.3592 0.4204 0.4544 0.5244 0.5864;
QTs 0.1418 0.6024 0.4283 0.3401 0.2921 0.2643 0.2355 0.1893 0.3480 0.3690 0.3838 0.4084 0.4638 0.5324 0.5946;QTo 0.2172 0.5828 0.4022 0.3015 0.2614 0.2190 0.1813 0.1546 0.3008 0.3466 0.3650 0.3842 0.4522 0.5104 0.5838;Q9s 0.2413 0.5746 0.4122 0.3308 0.2551 0.2173 0.1889 0.1793 0.3182 0.3540 0.3632 0.3834 0.4322 0.4952 0.5614;
Q9o 0.3122 0.5506 0.3579 0.3017 0.2296 0.1801 0.1520 0.1397 0.2734 0.3400 0.3396 0.3386 0.4082 0.4650 0.5602;Q8s 0.2745 0.5632 0.3793 0.2852 0.2454 0.2047 0.1887 0.1565 0.2844 0.3446 0.3582 0.3614 0.4106 0.4578 0.5564;Q8o 0.4103 0.5240 0.3577 0.2637 0.1918 0.1672 0.1528 0.1208 0.2536 0.3206 0.3180 0.3370 0.3910 0.4614 0.5546;
Q7s 0.4223 0.5182 0.3633 0.2708 0.2185 0.1889 0.1490 0.1583 0.2838 0.3308 0.3354 0.3560 0.3988 0.4784 0.5376;Q7o 0.3982 0.5246 0.3403 0.2436 0.1914 0.1462 0.1358 0.1115 0.2426 0.3014 0.2970 0.3230 0.3436 0.4478 0.5126;Q6s 0.3363 0.5468 0.3839 0.2740 0.2260 0.1838 0.1581 0.1444 0.2978 0.3028 0.3284 0.3672 0.3820 0.4496 0.5308;
Q6o 0.4434 0.5116 0.3115 0.2361 0.1685 0.1528 0.1243 0.1011 0.2392 0.2814 0.2844 0.3050 0.3598 0.4346 0.4972;Q5s 0.3680 0.5360 0.3483 0.2646 0.2105 0.1853 0.1646 0.1392 0.2878 0.3244 0.3202 0.3586 0.3858 0.4504 0.5318;Q5o 0.4992 0.5006 0.3208 0.2304 0.1534 0.1548 0.1209 0.0970 0.2482 0.2766 0.2912 0.3178 0.3316 0.4084 0.5064;
Q4s 0.4465 0.5110 0.3321 0.2522 0.1993 0.1629 0.1626 0.1313 0.2948 0.3108 0.3324 0.3396 0.3750 0.4458 0.5128;Q4o 0.5566 0.4810 0.3104 0.2253 0.1641 0.1295 0.1086 0.0901 0.2416 0.2930 0.2746 0.3150 0.3452 0.4004 0.4950;Q3s 0.5023 0.4988 0.3403 0.2452 0.1998 0.1649 0.1503 0.1339 0.2796 0.3006 0.3348 0.3410 0.3692 0.4242 0.5038;
Q3o 0.5234 0.4914 0.2867 0.1925 0.1461 0.1323 0.1020 0.0895 0.2326 0.2674 0.2772 0.3076 0.3270 0.3882 0.4836;Q2s 0.4570 0.5080 0.3213 0.2440 0.1914 0.1820 0.1430 0.1284 0.2774 0.3132 0.3152 0.3606 0.3652 0.3950 0.4870;Q2o 0.5324 0.4894 0.2927 0.1893 0.1549 0.1235 0.0994 0.0944 0.2384 0.2814 0.2984 0.3168 0.3300 0.3866 0.4752;
JJ 0.0181 0.7750 0.6141 0.4740 0.3936 0.3084 0.2859 0.2516 0.5296 0.5998 0.6170 0.6570 0.6872 0.7104 0.7746;JTs 0.2443 0.5732 0.4113 0.3512 0.2948 0.2397 0.2227 0.1987 0.3364 0.3526 0.3776 0.4064 0.4234 0.4798 0.5836;JTo 0.3333 0.5476 0.3981 0.3082 0.2617 0.2096 0.1912 0.1623 0.3070 0.3318 0.3438 0.3736 0.4076 0.4714 0.5682;
J9s 0.2474 0.5720 0.4005 0.3169 0.2571 0.2043 0.2015 0.1695 0.3320 0.3486 0.3744 0.3802 0.4220 0.4654 0.5560;J9o 0.3650 0.5372 0.3557 0.2783 0.2198 0.1849 0.1654 0.1343 0.2824 0.3120 0.3280 0.3558 0.3744 0.4368 0.5354;J8s 0.3891 0.5280 0.3608 0.2793 0.2396 0.2118 0.1825 0.1636 0.2900 0.3346 0.3586 0.3808 0.4046 0.4458 0.5376;
J8o 0.4193 0.5200 0.3394 0.2555 0.2047 0.1521 0.1464 0.1174 0.2390 0.3142 0.3208 0.3492 0.3776 0.4230 0.5230;J7s 0.4012 0.5244 0.3865 0.2841 0.2247 0.1914 0.1687 0.1390 0.2844 0.3186 0.3340 0.3592 0.3980 0.4296 0.5258;J7o 0.5113 0.4972 0.3177 0.2396 0.1775 0.1491 0.1194 0.1032 0.2544 0.2802 0.3120 0.3214 0.3482 0.3928 0.5044;
J6s 0.4600 0.5078 0.3294 0.2311 0.2015 0.1740 0.1574 0.1412 0.2528 0.3052 0.3202 0.3560 0.3526 0.3968 0.4982;J6o 0.5837 0.4722 0.3036 0.2210 0.1521 0.1295 0.1102 0.0972 0.2354 0.2548 0.2856 0.2968 0.3306 0.3650 0.4860;J5s 0.4630 0.5064 0.3349 0.2431 0.2052 0.1665 0.1525 0.1319 0.2790 0.3142 0.3212 0.3566 0.3520 0.4168 0.4966;
J5o 0.5415 0.4854 0.2787 0.2254 0.1532 0.1277 0.1112 0.0896 0.2276 0.2780 0.2976 0.3048 0.3124 0.3642 0.4834;J4s 0.5626 0.4780 0.3224 0.2491 0.2028 0.1679 0.1406 0.1243 0.2818 0.3062 0.3156 0.3496 0.3548 0.3864 0.4834;J4o 0.5928 0.4720 0.2838 0.2089 0.1539 0.1262 0.1010 0.0844 0.2256 0.2654 0.2814 0.3016 0.3374 0.3708 0.4602;
J3s 0.5596 0.4792 0.2946 0.2386 0.1920 0.1525 0.1466 0.1318 0.2556 0.3144 0.3186 0.3400 0.3504 0.3938 0.4884;J3o 0.6621 0.4438 0.2637 0.1886 0.1603 0.1313 0.1006 0.0898 0.2318 0.2520 0.2792 0.3114 0.3070 0.3498 0.4466;J2s 0.5656 0.4740 0.3102 0.2308 0.1657 0.1546 0.1520 0.1321 0.2572 0.2842 0.3202 0.3440 0.3650 0.3904 0.4732;
J2o 0.6410 0.4584 0.2549 0.1852 0.1461 0.1211 0.0937 0.0767 0.2384 0.2534 0.2756 0.3034 0.3030 0.3472 0.4288;TT 0.0226 0.7532 0.5769 0.4496 0.3639 0.2986 0.2522 0.2184 0.4470 0.5502 0.5654 0.6128 0.6764 0.7224 0.7468;T9s 0.3529 0.5406 0.3815 0.3168 0.2608 0.2246 0.2036 0.1703 0.3106 0.3720 0.3732 0.3836 0.4052 0.4478 0.5310;
T9o 0.4902 0.5016 0.3581 0.2771 0.2227 0.1951 0.1659 0.1320 0.2798 0.3472 0.3504 0.3394 0.3736 0.4134 0.5086;T8s 0.4253 0.5170 0.3696 0.2733 0.2337 0.2083 0.1805 0.1651 0.2978 0.3364 0.3746 0.3752 0.3892 0.4200 0.5228;T8o 0.4811 0.5028 0.3338 0.2515 0.2161 0.1674 0.1374 0.1275 0.2566 0.3314 0.3226 0.3356 0.3644 0.4112 0.4924;
T7s 0.4495 0.5108 0.3422 0.2756 0.2038 0.1839 0.1677 0.1425 0.2648 0.3338 0.3364 0.3566 0.3876 0.4184 0.4898;T7o 0.6048 0.4666 0.3014 0.2242 0.1914 0.1558 0.1143 0.1055 0.2380 0.2712 0.3050 0.3396 0.3638 0.3790 0.4970;T6s 0.5445 0.4834 0.3376 0.2389 0.2015 0.1795 0.1518 0.1368 0.2724 0.3042 0.3236 0.3392 0.3856 0.3934 0.4878;
T6o 0.6320 0.4614 0.2913 0.2192 0.1555 0.1324 0.1132 0.0879 0.2478 0.2868 0.3034 0.3198 0.3366 0.3682 0.4584;T5s 0.6229 0.4618 0.2931 0.2216 0.1847 0.1641 0.1476 0.1242 0.2574 0.3086 0.2962 0.3362 0.3412 0.3688 0.4870;T5o 0.7315 0.4302 0.2779 0.1798 0.1437 0.1258 0.0879 0.0779 0.2162 0.2718 0.2788 0.2960 0.3264 0.3366 0.4438;
T4s 0.6078 0.4666 0.3116 0.2233 0.2049 0.1586 0.1390 0.1268 0.2682 0.3062 0.3206 0.3386 0.3482 0.3712 0.4534;T4o 0.6712 0.4426 0.2662 0.2012 0.1379 0.1177 0.0874 0.0850 0.2238 0.2724 0.2872 0.3100 0.3186 0.3340 0.4692;T3s 0.6501 0.4480 0.2849 0.2175 0.1731 0.1657 0.1260 0.1028 0.2636 0.3170 0.3218 0.3498 0.3530 0.3726 0.4712;
T3o 0.7104 0.4380 0.2569 0.1829 0.1174 0.1238 0.1010 0.0767 0.2330 0.2664 0.2698 0.2912 0.3138 0.3314 0.4282;T2s 0.6742 0.4412 0.2795 0.2219 0.1829 0.1516 0.1221 0.1195 0.2506 0.3152 0.3094 0.3464 0.3624 0.3462 0.4466;T2o 0.7436 0.4238 0.2433 0.1764 0.1239 0.1102 0.0879 0.0803 0.2218 0.2692 0.2726 0.3020 0.3144 0.3378 0.3900;
99 0.0271 0.7164 0.5003 0.3997 0.3316 0.2689 0.2290 0.1813 0.3870 0.5126 0.5328 0.5716 0.6246 0.6576 0.7274;98s 0.5143 0.4922 0.3743 0.2880 0.2333 0.1972 0.1908 0.1410 0.2722 0.3564 0.3738 0.3696 0.4150 0.4188 0.4912;98o 0.5747 0.4738 0.3417 0.2461 0.1963 0.1718 0.1461 0.1249 0.2408 0.3090 0.3140 0.3650 0.3598 0.3872 0.4854;
97s 0.5475 0.4824 0.3456 0.2717 0.2310 0.2019 0.1589 0.1511 0.2878 0.3290 0.3382 0.3726 0.3960 0.4070 0.5016;97o 0.6199 0.4622 0.3152 0.2311 0.1759 0.1482 0.1268 0.1123 0.2540 0.3222 0.3054 0.3388 0.3380 0.3626 0.4716;96s 0.5958 0.4712 0.3214 0.2537 0.2028 0.1689 0.1442 0.1429 0.2688 0.3318 0.3278 0.3726 0.3848 0.3824 0.4748;
96o 0.7014 0.4402 0.2769 0.2063 0.1531 0.1290 0.1080 0.0906 0.2364 0.2906 0.3118 0.3224 0.3548 0.3542 0.4178;95s 0.6440 0.4534 0.3053 0.2272 0.2001 0.1505 0.1357 0.1190 0.2548 0.3074 0.3246 0.3488 0.3612 0.3616 0.4694;95o 0.7677 0.4168 0.2761 0.1709 0.1470 0.1322 0.1041 0.0775 0.2140 0.2710 0.2892 0.2932 0.3254 0.3510 0.4250;
94s 0.7345 0.4248 0.2935 0.2008 0.1703 0.1473 0.1118 0.1099 0.2416 0.2918 0.3098 0.3400 0.3398 0.3662 0.4392;94o 0.8160 0.4010 0.2369 0.1707 0.1413 0.1059 0.0851 0.0706 0.2190 0.2586 0.2774 0.2986 0.3200 0.3148 0.4058;93s 0.7225 0.4308 0.2778 0.2076 0.1911 0.1275 0.1324 0.1131 0.2468 0.2996 0.3166 0.3196 0.3488 0.3604 0.4486;
93o 0.8552 0.3908 0.2469 0.1581 0.1280 0.0980 0.0880 0.0720 0.2156 0.2596 0.2824 0.2930 0.3162 0.3156 0.4006;92s 0.7858 0.4104 0.2850 0.2086 0.1690 0.1384 0.1239 0.1057 0.2366 0.2852 0.3054 0.3346 0.3480 0.3662 0.4134;92o 0.8250 0.4006 0.2395 0.1560 0.1296 0.0930 0.0842 0.0648 0.1964 0.2570 0.2664 0.2752 0.2876 0.3012 0.4120;
88 0.0317 0.6840 0.4915 0.3702 0.2914 0.2326 0.2139 0.1780 0.3150 0.4882 0.5174 0.5506 0.5998 0.6364 0.6882;87s 0.6109 0.4654 0.3473 0.2732 0.2184 0.1979 0.1684 0.1556 0.2816 0.3344 0.3590 0.3704 0.3884 0.4134 0.4934;87o 0.6833 0.4406 0.2987 0.2251 0.1830 0.1526 0.1214 0.1206 0.2294 0.2858 0.3318 0.3340 0.3754 0.3528 0.4572;
86s 0.6531 0.4448 0.3005 0.2401 0.2011 0.1680 0.1588 0.1439 0.2604 0.3270 0.3462 0.3752 0.4108 0.3980 0.4740;86o 0.6923 0.4406 0.2881 0.2248 0.1751 0.1409 0.1249 0.1043 0.2374 0.2792 0.3194 0.3322 0.3646 0.3516 0.4426;85s 0.7134 0.4368 0.2906 0.2329 0.1824 0.1587 0.1340 0.1253 0.2508 0.3242 0.3294 0.3340 0.3840 0.3870 0.4400;
85o 0.7526 0.4234 0.2423 0.2004 0.1497 0.1288 0.0952 0.0880 0.2256 0.2780 0.3068 0.3098 0.3430 0.3406 0.4050;84s 0.7164 0.4336 0.2787 0.2118 0.1746 0.1437 0.1268 0.1199 0.2460 0.2840 0.3066 0.3324 0.3614 0.3730 0.4360;84o 0.8341 0.3942 0.2329 0.1794 0.1261 0.1017 0.0972 0.0830 0.2088 0.2696 0.3136 0.3020 0.3296 0.3242 0.3982;
83s 0.7919 0.4074 0.2696 0.2023 0.1635 0.1509 0.1266 0.1149 0.2236 0.2866 0.3062 0.3416 0.3344 0.3454 0.4000;83o 0.8854 0.3782 0.2145 0.1451 0.1096 0.0930 0.0736 0.0687 0.1900 0.2488 0.2944 0.2970 0.2838 0.3202 0.3732;82s 0.8069 0.4048 0.2544 0.1995 0.1759 0.1385 0.1237 0.1066 0.2272 0.2882 0.2946 0.3146 0.3320 0.3506 0.3936;
82o 0.9065 0.3712 0.1949 0.1590 0.1078 0.0884 0.0772 0.0700 0.1744 0.2530 0.2818 0.2892 0.2972 0.3076 0.3570;77 0.0513 0.6568 0.4657 0.3489 0.2567 0.2238 0.1796 0.1655 0.2978 0.4218 0.4866 0.5426 0.5700 0.5886 0.6376;76s 0.6471 0.4528 0.3054 0.2257 0.2113 0.1829 0.1562 0.1382 0.2754 0.3290 0.3370 0.3784 0.3812 0.3812 0.4604;
76o 0.7768 0.4162 0.2918 0.2025 0.1820 0.1427 0.1093 0.1087 0.2268 0.2964 0.3026 0.3354 0.3606 0.3828 0.4230;75s 0.7195 0.4330 0.2961 0.2302 0.1981 0.1669 0.1390 0.1346 0.2554 0.3068 0.3366 0.3538 0.3758 0.3790 0.4296;75o 0.8431 0.3930 0.2709 0.1998 0.1571 0.1356 0.1136 0.1058 0.2196 0.2976 0.3126 0.3164 0.3376 0.3572 0.4074;
74s 0.7798 0.4152 0.2791 0.2287 0.1820 0.1599 0.1391 0.1317 0.2472 0.3038 0.3280 0.3404 0.3720 0.3796 0.4212;74o 0.8763 0.3858 0.2500 0.1817 0.1356 0.1155 0.0981 0.0768 0.2094 0.2704 0.2952 0.3132 0.3288 0.3290 0.3830;73s 0.8582 0.3908 0.2743 0.2084 0.1599 0.1337 0.1232 0.1098 0.2352 0.2886 0.3128 0.3372 0.3598 0.3318 0.3934;
73o 0.9457 0.3620 0.2143 0.1590 0.1162 0.0939 0.0932 0.0777 0.1986 0.2598 0.2736 0.2930 0.3028 0.3302 0.3602;72s 0.8884 0.3748 0.2323 0.1982 0.1519 0.1293 0.1189 0.1045 0.2164 0.2792 0.3088 0.3204 0.3316 0.3508 0.3802;72o 0.9638 0.3584 0.2057 0.1412 0.0998 0.0866 0.0634 0.0722 0.1798 0.2280 0.2594 0.2842 0.3058 0.3228 0.3602;
66 0.0860 0.6246 0.4359 0.3015 0.2359 0.1952 0.1784 0.1487 0.2996 0.3874 0.4524 0.4898 0.5452 0.5624 0.6308;65s 0.7557 0.4228 0.2920 0.2394 0.1866 0.1753 0.1559 0.1281 0.2752 0.3130 0.3474 0.3506 0.3854 0.3884 0.4080;65o 0.8009 0.4072 0.2779 0.1911 0.1484 0.1388 0.1191 0.1025 0.2464 0.2844 0.2994 0.3298 0.3334 0.3632 0.4054;
64s 0.7828 0.4124 0.2720 0.2196 0.1813 0.1609 0.1395 0.1360 0.2658 0.3114 0.3276 0.3606 0.3576 0.3700 0.4250;64o 0.8673 0.3896 0.2403 0.1821 0.1483 0.1196 0.0986 0.0905 0.2158 0.2738 0.2766 0.3208 0.3248 0.3514 0.3872;63s 0.7888 0.4082 0.2757 0.1952 0.1735 0.1348 0.1294 0.1195 0.2448 0.3124 0.3200 0.3466 0.3466 0.3586 0.3886;
63o 0.9336 0.3656 0.2334 0.1738 0.1237 0.1117 0.0822 0.0768 0.2082 0.2614 0.2722 0.3156 0.3164 0.3350 0.3556;62s 0.9367 0.3648 0.2375 0.1735 0.1482 0.1198 0.1222 0.1064 0.2396 0.2974 0.2952 0.3288 0.3504 0.3644 0.3670;62o 0.9910 0.3386 0.2051 0.1441 0.1052 0.0916 0.0794 0.0680 0.2126 0.2426 0.2772 0.2966 0.3040 0.3098 0.3442;
55 0.1026 0.6134 0.3854 0.3016 0.2258 0.1745 0.1497 0.1410 0.2898 0.4118 0.4260 0.4854 0.5260 0.5310 0.6052;54s 0.7587 0.4176 0.3008 0.2382 0.1941 0.1638 0.1479 0.1396 0.2854 0.3186 0.3560 0.3638 0.3770 0.3924 0.4120;54o 0.8974 0.3748 0.2631 0.1833 0.1462 0.1228 0.1125 0.0998 0.2546 0.2862 0.3324 0.3284 0.3456 0.3400 0.3790;
53s 0.8039 0.4054 0.2765 0.2213 0.1783 0.1444 0.1406 0.1249 0.2538 0.3062 0.3244 0.3668 0.3646 0.3638 0.3996;53o 0.9186 0.3698 0.2353 0.1717 0.1288 0.1109 0.0977 0.0938 0.2244 0.2916 0.3022 0.3032 0.3148 0.3418 0.3564;52s 0.8462 0.3914 0.2551 0.1872 0.1710 0.1398 0.1341 0.1150 0.2400 0.2906 0.3138 0.3484 0.3612 0.3680 0.3698;
52o 0.9729 0.3484 0.2152 0.1538 0.1193 0.0971 0.0854 0.0804 0.2134 0.2494 0.2810 0.3048 0.3102 0.3164 0.3300;44 0.2821 0.5616 0.3641 0.2695 0.1936 0.1734 0.1553 0.1382 0.2976 0.3690 0.4210 0.4586 0.5038 0.5360 0.5808;43s 0.9095 0.3708 0.2746 0.2158 0.1807 0.1553 0.1308 0.1221 0.2632 0.2976 0.3080 0.3558 0.3660 0.3786 0.4118;
43o 0.9548 0.3594 0.2274 0.1595 0.1263 0.1181 0.0912 0.0764 0.2264 0.2690 0.2804 0.3188 0.3238 0.3450 0.3502;42s 0.9216 0.3666 0.2223 0.1818 0.1620 0.1428 0.1183 0.1021 0.2448 0.2944 0.3130 0.3450 0.3442 0.3448 0.3762;42o 0.9819 0.3414 0.2094 0.1427 0.1277 0.0899 0.0880 0.0799 0.1936 0.2534 0.2752 0.2968 0.3076 0.3084 0.3218;
33 0.3499 0.5430 0.3437 0.2359 0.2016 0.1469 0.1385 0.1276 0.2660 0.3898 0.4274 0.4540 0.5018 0.5078 0.5412;32s 0.9246 0.3666 0.2325 0.1767 0.1578 0.1334 0.1140 0.1142 0.2666 0.2842 0.3072 0.3342 0.3480 0.3270 0.3642;32o 1.0000 0.3326 0.2009 0.1372 0.1074 0.0907 0.0762 0.0737 0.2134 0.2508 0.2752 0.2962 0.2838 0.3272 0.3292;
22 0.4540 0.5096 0.3177 0.2138 0.1828 0.1528 0.1484 0.1254 0.2946 0.3838 0.4098 0.4520 0.4646 0.4944 0.5124"""

_MULTI = {}   # class -> (eq vs 1 random, ..., vs 7 random)
_RANGE = {}   # class -> HU eq vs top-r range for r in _RANGES
_PCT = {}     # class -> cumulative combo share of hands at least as strong (0..1)
for _line in _TABLE_TEXT.split(";"):
    _p = _line.split()
    if not _p:
        continue
    _c = _p[0]
    _v = [float(x) for x in _p[1:]]
    _PCT[_c] = _v[0]
    _MULTI[_c] = tuple(_v[1:8])
    _RANGE[_c] = tuple(_v[8:15])

_RCH = "23456789TJQKA"
_RV = {r: i for i, r in enumerate(_RCH)}
_FULL_DECK = tuple(range(52))


def _cls(hole):
    a, b = hole[0], hole[1]
    ra, rb = _RV[a[0]], _RV[b[0]]
    if ra < rb:
        ra, rb = rb, ra
    if ra == rb:
        return _RCH[ra] + _RCH[rb]
    return _RCH[ra] + _RCH[rb] + ("s" if a[1] == b[1] else "o")


def _range_eq(c, w):
    """HU equity of class c vs a top-w range, linear interpolation over _RANGES."""
    row = _RANGE[c]
    rs = _RANGES
    if w <= rs[0]:
        return row[0]
    for k in range(1, len(rs)):
        if w <= rs[k]:
            t = (w - rs[k - 1]) / (rs[k] - rs[k - 1])
            return row[k - 1] + t * (row[k] - row[k - 1])
    return row[-1]


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Bot(Agent):
    MC_BUDGET_NS = 300_000      # act() MC deadline (wall clock), leaves >0.6 ms headroom under the 1 ms rule
    WARM_BUDGET_NS = 250_000    # observe() warm-up per street
    MC_CAP = 900                # stop sampling a board once this many samples are cached

    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.n = config.get("n_seats", 8)
        self.rng = random.Random(config.get("seed", 0))
        self.icm = make_icm(config.get("payouts") or (50, 30, 20, 0, 0, 0, 0, 0))
        # opponent model: counts with priors
        n = self.n
        self.hands = [0] * n
        self.vpip = [0] * n
        self.pfr = [0] * n
        self.fb_face = [0] * n      # postflop: times facing a bet
        self.fb_fold = [0] * n
        self._vp_hand = [False] * n
        self._pr_hand = [False] * n
        self._bets = [0] * n
        self._street = "preflop"
        self.hole = None
        self.mc = {}                # board key -> [wins, samples]
        self.preflop_aggr = False

    # ------------------------------------------------------------------ observe: opponent model + MC warm-up
    def observe(self, ev):
        t = ev.get("type")
        if t == "action":
            s = ev["seat"]
            a = ev["action"]
            if self._street == "preflop":
                if a in ("call", "raise") and not self._vp_hand[s]:
                    self._vp_hand[s] = True
                    self.vpip[s] += 1
                if a == "raise" and not self._pr_hand[s]:
                    self._pr_hand[s] = True
                    self.pfr[s] += 1
            else:
                facing = max(self._bets) > self._bets[s]
                if facing and a in ("fold", "call", "raise"):
                    self.fb_face[s] += 1
                    if a == "fold":
                        self.fb_fold[s] += 1
            if a in ("call", "raise"):
                self._bets[s] = ev["amount"]
        elif t == "street":
            self._street = ev["street"]
            self._bets = [0] * self.n
            if self.hole is not None:
                self._warm(ev["board"])
        elif t == "hand_start":
            self._street = "preflop"
            self._bets = [0] * self.n
            for p in ev.get("posts", ()) or ():
                try:
                    s, kind, amt = p
                    if kind != "ante":
                        self._bets[s] = amt
                except Exception:
                    pass
            for i, st in enumerate(ev.get("stacks", ())):
                if st > 0:
                    self.hands[i] += 1
            self._vp_hand = [False] * self.n
            self._pr_hand = [False] * self.n
            self.mc = {}
            self.hole = None
            self.preflop_aggr = False
        elif t == "deal":
            self.hole = tuple(ev["hole"])

    def _warm(self, board):
        try:
            self._mc_equity(self.hole, tuple(board), _perf() + self.WARM_BUDGET_NS)
        except Exception:
            pass

    # ------------------------------------------------------------------ stats with priors
    def _vpip(self, s):
        return (self.vpip[s] + 3.0) / (self.hands[s] + 10.0)

    def _pfr(self, s):
        return (self.pfr[s] + 1.5) / (self.hands[s] + 10.0)

    def _fold_to_bet(self, s):
        return (self.fb_fold[s] + 3.2) / (self.fb_face[s] + 8.0)

    # ------------------------------------------------------------------ postflop Monte Carlo (HU vs random hand)
    def _mc_equity(self, hole, board, deadline):
        key = board
        rec = self.mc.get(key)
        if rec is None:
            rec = self.mc[key] = [0.0, 0]
        if rec[1] >= self.MC_CAP:
            return rec[0] / rec[1]
        h = [CARD_INT[c] for c in hole]
        b = [CARD_INT[c] for c in board]
        used = set(h) | set(b)
        deck = [c for c in _FULL_DECK if c not in used]
        need = 5 - len(b)
        sample = self.rng.sample
        wins = 0.0
        cnt = 0
        k_opp = 3   # opponent hands per sampled runout (re-uses my evaluation)
        while True:
            d = sample(deck, need + 2 * k_opp)
            full = b + d[:need]
            me = evaluate(h + full)
            for j in range(k_opp):
                o = evaluate(d[need + 2 * j: need + 2 * j + 2] + full)
                if me > o:
                    wins += 1.0
                elif me == o:
                    wins += 0.5
            cnt += k_opp
            if cnt + rec[1] >= self.MC_CAP or _perf() > deadline:
                break
        rec[0] += wins
        rec[1] += cnt
        return rec[0] / rec[1]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _raise(state, target):
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        t = int(target)
        return {"action": "raise", "amount": lo if t < lo else hi if t > hi else t}

    @staticmethod
    def _passive(state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}

    @staticmethod
    def _shove(state):
        legal = state["legal_actions"]
        if "raise" in legal:
            return {"action": "raise", "amount": state["max_raise_to"]}
        if "call" in legal:
            return {"action": "call"}
        return {"action": "check"}

    def _icm_vec(self, state):
        # chips each seat would have if the hand ended now with everyone taking back what they put in
        st = state["stacks"]
        co = state["contributed"]
        return [st[i] + co[i] if state["alive"][i] else 0 for i in range(len(st))]

    def _allin_icm_ev(self, state, eq, p_fold, opp):
        """ICM points of going all-in (vs folding/checking now) against one likely caller `opp`.
        Returns (ev_allin, ev_fold)."""
        me = self.seat
        base = self._icm_vec(state)
        my_in = state["contributed"][me]
        my_total = base[me]
        opp_total = base[opp] if opp is not None else 0
        dead = state["pot"] - my_in - (state["contributed"][opp] if opp is not None else 0)
        # fold: I lose what I already put in; dead money goes (roughly) to the opponent
        f = list(base)
        f[me] = my_total - my_in
        if opp is not None:
            f[opp] = opp_total + my_in
        ev_fold = self.icm(f, me)
        # everyone folds: I win the dead money and opp's contribution
        w0 = list(base)
        w0[me] = my_total + dead + (state["contributed"][opp] if opp is not None else 0)
        if opp is not None:
            w0[opp] = opp_total - state["contributed"][opp]
        ev_steal = self.icm(w0, me)
        if opp is None:
            return ev_steal, ev_fold
        risk = min(my_total, opp_total)
        win = list(base)
        win[me] = my_total + risk + dead
        win[opp] = opp_total - risk
        lose = list(base)
        lose[me] = my_total - risk
        lose[opp] = opp_total + risk + dead
        ev_called = eq * self.icm(win, me) + (1.0 - eq) * self.icm(lose, me)
        return p_fold * ev_steal + (1.0 - p_fold) * ev_called, ev_fold

    # ------------------------------------------------------------------ act
    def act(self, state):
        try:
            a = self._decide(state)
            k, _, err = validate_action(a, state["legal_actions"], state["min_raise_to"], state["max_raise_to"])
            if err is None:
                return a
        except Exception:
            pass
        try:
            return self._passive(state)
        except Exception:
            return {"action": "fold"}

    def _decide(self, state):
        if state["street"] == "preflop":
            return self._preflop(state)
        return self._postflop(state)

    # ------------------------------------------------------------------ preflop
    def _preflop(self, state):
        me = self.seat
        n = state["n_seats"]
        bb = max(1, state["bb"])
        legal = state["legal_actions"]
        stacks, bets, active, all_in = state["stacks"], state["bets"], state["active"], state["all_in"]
        c = _cls(state["hole"])
        pct = _PCT[c]
        my_total = stacks[me] + bets[me]
        others = [i for i in range(n) if i != me and active[i]]
        live_others = [i for i in others if not all_in[i]]
        opp_max = max((stacks[i] + bets[i] for i in others), default=0)
        eff = min(my_total, opp_max)
        m = eff / bb
        to_call = state["to_call"]
        top = max(bets)
        hist = state["history_this_hand"]
        raisers = [s for (s, a, _) in hist if a == "raise"]
        n_raises = len(raisers)
        pot = state["pot"]

        # players still to act behind me this round (unopened pot: up to and including the big blind)
        behind = []
        j = me
        for _ in range(n - 1):
            j = (j + 1) % n
            if active[j] and not all_in[j]:
                behind.append(j)
            if j == state["bb_seat"]:
                break

        if n_raises == 0:
            # ---------- unopened (maybe limped)
            if m <= 15 and "raise" in legal:
                # push/fold in ICM points; callers call a top-w range, wider vs short shoves
                tot_fold = 1.0
                caller = None
                wmax = 0.0
                for s in behind:
                    w = _clip(1.7 / max(m, 1.0), 0.07, 0.55) * _clip(self._vpip(s) / 0.30, 0.6, 1.8)
                    w = _clip(w, 0.05, 0.75)
                    tot_fold *= (1.0 - w)
                    if w > wmax or caller is None:
                        wmax = w
                        caller = s
                if caller is None:
                    caller = others[0] if others else None
                    wmax = 0.5
                eq = _range_eq(c, wmax)
                ev_ai, ev_f = self._allin_icm_ev(state, eq, tot_fold, caller)
                if "check" in legal:   # BB option: compare against checking (keep the hand)
                    ev_f = self.icm(self._icm_vec(state), me) + 0.0
                if ev_ai > ev_f:
                    return self._shove(state)
                return self._passive(state) if "check" in legal else (
                    {"action": "call"} if to_call <= bb // 2 and pct <= 0.35 and "call" in legal else self._passive(state))
            nb = len(behind)
            open_w = 0.95 / (nb + 1.0) + (0.05 if nb <= 2 else 0.0)
            limpers = sum(1 for (s, a, _) in hist if a == "call")
            if pct <= open_w and "raise" in legal:
                self.preflop_aggr = True
                return self._raise(state, int(2.5 * bb + limpers * bb))
            if "check" in legal:
                return {"action": "check"}
            if "call" in legal and to_call <= bb // 2 and pct <= 0.45:
                return {"action": "call"}
            if "call" in legal and limpers >= 1 and to_call <= bb and pct <= 0.25:
                return {"action": "call"}
            return self._passive(state)

        # ---------- facing a raise
        last = raisers[-1]
        raiser_total = stacks[last] + bets[last]
        raiser_allin = all_in[last]
        size_bb = top / bb
        w_r = self._pfr(last) * (1.0 if size_bb <= 3.5 else 0.75)
        if n_raises >= 2:
            w_r *= 0.4
        if raiser_allin:
            w_r = max(w_r, _clip(1.4 / max(raiser_total / bb, 1.0), 0.05, 0.5))
        w_r = _clip(w_r, 0.025, 0.6)
        eq = _range_eq(c, w_r)
        extra = len([i for i in others if i != last and bets[i] > 0 and active[i]])
        if extra:
            eq = eq ** (1.0 + 0.7 * extra)

        commit = to_call >= 0.4 * stacks[me] or m <= 20 or raiser_allin
        if commit:
            # all-in decision (shove or call off) in ICM points
            if "raise" in legal and not raiser_allin:
                p_fold = _clip(1.0 - (0.55 if n_raises == 1 else 0.75), 0.0, 1.0)
                p_fold *= _clip(1.0 - (eq - 0.5), 0.5, 1.0) if pct > w_r else 1.0
                ev_ai, ev_f = self._allin_icm_ev(state, eq, p_fold, last)
                if ev_ai > ev_f:
                    return self._shove(state)
            if "call" in legal:
                ev_call, ev_f = self._allin_icm_ev(state, eq, 0.0, last)
                if ev_call > ev_f:
                    return {"action": "call"} if to_call >= stacks[me] or "raise" not in legal else self._shove(state)
            return self._passive(state)

        # deep: 3-bet for value, call with pot odds plus a realization haircut, else fold
        if "raise" in legal and (pct <= 0.025 or pct <= w_r * 0.3):
            self.preflop_aggr = True
            return self._raise(state, int(3 * top + sum(bets) - top))
        po = to_call / float(pot + to_call)
        realize = 0.82 if state["button"] != me else 0.92
        if "call" in legal and eq * realize > po + 0.03 and to_call <= 0.25 * stacks[me]:
            return {"action": "call"}
        return self._passive(state)

    # ------------------------------------------------------------------ postflop
    def _postflop(self, state):
        t0 = _perf()
        me = self.seat
        n = state["n_seats"]
        legal = state["legal_actions"]
        stacks, bets, active = state["stacks"], state["bets"], state["active"]
        pot = state["pot"]
        to_call = state["to_call"]
        board = tuple(state["board"])
        opps = [i for i in range(n) if i != me and active[i]]
        n_opp = max(1, len(opps))
        e_hu = self._mc_equity(tuple(state["hole"]), board, t0 + self.MC_BUDGET_NS)
        # flop-seeing ranges are stronger than random: mild per-opponent discount, then multiway product
        e = e_hu ** (1.15 * n_opp)
        my_stack = stacks[me]
        top = max(bets)
        river = len(board) == 5

        if to_call > 0:
            pre_pot = max(1, pot - to_call)  # pot before the bet facing us (roughly)
            e_adj = e ** (1.0 + 1.1 * min(2.0, to_call / pre_pot))
            po = to_call / float(pot + to_call)
            if "raise" in legal and e_adj > 0.70:
                tgt = top + int(1.0 * (pot + to_call))
                if tgt >= 0.6 * (my_stack + bets[me]):
                    return self._shove(state)
                return self._raise(state, tgt)
            margin = 0.0 if river else -0.02   # small implied-odds credit before the river
            if e_adj >= po + margin and "call" in legal:
                return {"action": "call"}
            return self._passive(state)

        # nobody bet: bet vs check by EV with fold equity
        if "raise" not in legal:
            return {"action": "check"}
        f_all = 1.0
        for s in opps:
            f_all *= self._fold_to_bet(s)
        b = int(0.6 * pot) if e < 0.8 else int(0.75 * pot)
        b = max(b, state["min_raise_to"])
        if b >= 0.7 * my_stack:
            b = my_stack + bets[me]
        e_called = e ** 2.0 if e < 0.75 else e ** 1.2
        in_pos = all(((s - state["button"] - 1) % n) < ((me - state["button"] - 1) % n) for s in opps)
        realize = 0.95 if in_pos else 0.8
        ev_check = e * pot * realize
        ev_bet = f_all * pot + (1.0 - f_all) * (e_called * (pot + 2 * b) - b)
        if not self.preflop_aggr and e < 0.55:
            ev_bet -= 0.1 * pot   # no c-bet credibility: shrink bluffs when we did not lead preflop
        if ev_bet > ev_check:
            return self._raise(state, b)
        return {"action": "check"}


# ---------------------------------------------------------------------- tiny timing check (not used in play)
if __name__ == "__main__":
    # .venv/bin/python sim/field_ai/ai_quant.py [tables]   -> act() CPU-time percentiles vs the 1 ms limit
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from sim.kit.runner import run
    n_tab = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load", "%.2f %.2f %.2f" % os.getloadavg())
    for gc_mode in ("default", "disable"):
        res = run([("me", "sim/field_ai/ai_quant.py:Bot")] * 8, n_tab, seed=7, workers=1, timing="cpu",
                  gc_mode=gc_mode)
        s = res.entrants[0].stats.summary()
        tot = {"dec": 0, "over": 0, "exc": 0, "ill": 0, "max": 0.0}
        for e in res.entrants:
            ss = e.stats.summary()
            tot["dec"] += ss["decisions"]
            tot["over"] += ss["over_limit"]["cpu"]
            tot["exc"] += ss["exceptions"]
            tot["ill"] += ss["illegal"]
            tot["max"] = max(tot["max"], ss["cpu_us"]["max"])
        cpu = s["cpu_us"]
        print(f"gc={gc_mode:8s} seat0 cpu_us p50={cpu['p50']:.1f} p99={cpu['p99']:.1f} p99.9={cpu['p99.9']:.1f} "
              f"max={cpu['max']:.1f} | all seats: decisions={tot['dec']} cpu_over_1ms={tot['over']} "
              f"exc={tot['exc']} illegal={tot['ill']} max_us={tot['max']:.1f}")
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load", "%.2f %.2f %.2f" % os.getloadavg(),
          "PASS" if tot["max"] < 1000 else "CHECK max")
