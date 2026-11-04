"""ai_frontier_careful: a careful, table-driven 8-max SNG bot (team persona: two people, best model, one hour).

Strategy summary (why these choices, given the rules):
- Format: 8 seats, 2,000 chips, blinds up every 25 hands, 200-hand cap with an ICM split, top-heavy points, six
  tournaments weighted 1..6, 1 ms per decision, errors auto-fold. So: (a) never error, never be slow; (b) survival
  has real value (top-heavy payouts), chip accumulation only matters when it converts to places; (c) by hand ~100
  everyone is 10-20 bb deep, so most of the table is push/fold poker; (d) the field is mostly AI-written bots built
  in an hour: tight-passive, fold too much to preflop raises, rarely bluff, call down with any pair, and a share of
  them are broken (always fold / check). Against that field, relentless but hand-selected aggression (steal blinds,
  c-bet once, value-bet thinly) beats "play solid".
- Preflop: 169-class tables precomputed OFFLINE with the kit evaluator (equity vs 1 and 2 random hands, equity vs
  the top 8/15/25/40/60% of hands, and a combo-weighted percentile). Deep (> 16 bb): position-sized open ranges,
  3-bet premiums, cheap calls with pairs/suited broadways. Short (<= 16 bb or facing a shove): push/fold with a
  Nash-shaped shove chart (threshold by players behind and stack in bb). Shove CALLS use exact ICM (kit icm.make_icm)
  on win/lose/fold stack vectors with equity against an estimated range; that is where ICM matters most and the
  cost (~0.1-0.2 ms CPU) is affordable only there.
- Postflop: cheap made-hand / draw classification from rank masks plus the kit evaluator; value bet 55-65% pot,
  one c-bet as the aggressor heads-up, fold without a hand or a draw, call draws by pot odds, do not pay off big
  bets with one pair against non-maniacs, never bluff-raise.
- Opponent model (observe(), untimed): per-seat VPIP, PFR, fold-to-preflop-raise, fold-to-postflop-bet. Used to
  widen steals against folders, tighten shove-calls against tight shovers, and skip c-bets against stations.
- Safety: every act() runs inside try/except, the result is validated with the kit's validate_action, and any
  problem returns the engine's own fallback (check if free, else fold). All raises are clamped to the legal bounds.
  No per-decision simulation; everything is table lookups and small arithmetic.
"""
from sim.kit.agent import Agent, validate_action
from sim.kit.cards import CARD_INT, evaluate
from sim.kit.icm import make_icm

# == TABLES ==  (generated offline with sim.kit.cards.evaluate; see module docstring)
# HAND_NAMES, EQ1, EQ2, PCT, RANGE_CUTS, EQR are spliced in below this marker.
HAND_NAMES = ('AA', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo', 'A9s', 'A9o', 'A8s', 'A8o', 'A7s', 'A7o', 'A6s', 'A6o', 'A5s', 'A5o', 'A4s', 'A4o', 'A3s', 'A3o', 'A2s', 'A2o', 'KK', 'KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'KTo', 'K9s', 'K9o', 'K8s', 'K8o', 'K7s', 'K7o', 'K6s', 'K6o', 'K5s', 'K5o', 'K4s', 'K4o', 'K3s', 'K3o', 'K2s', 'K2o', 'QQ', 'QJs', 'QJo', 'QTs', 'QTo', 'Q9s', 'Q9o', 'Q8s', 'Q8o', 'Q7s', 'Q7o', 'Q6s', 'Q6o', 'Q5s', 'Q5o', 'Q4s', 'Q4o', 'Q3s', 'Q3o', 'Q2s', 'Q2o', 'JJ', 'JTs', 'JTo', 'J9s', 'J9o', 'J8s', 'J8o', 'J7s', 'J7o', 'J6s', 'J6o', 'J5s', 'J5o', 'J4s', 'J4o', 'J3s', 'J3o', 'J2s', 'J2o', 'TT', 'T9s', 'T9o', 'T8s', 'T8o', 'T7s', 'T7o', 'T6s', 'T6o', 'T5s', 'T5o', 'T4s', 'T4o', 'T3s', 'T3o', 'T2s', 'T2o', '99', '98s', '98o', '97s', '97o', '96s', '96o', '95s', '95o', '94s', '94o', '93s', '93o', '92s', '92o', '88', '87s', '87o', '86s', '86o', '85s', '85o', '84s', '84o', '83s', '83o', '82s', '82o', '77', '76s', '76o', '75s', '75o', '74s', '74o', '73s', '73o', '72s', '72o', '66', '65s', '65o', '64s', '64o', '63s', '63o', '62s', '62o', '55', '54s', '54o', '53s', '53o', '52s', '52o', '44', '43s', '43o', '42s', '42o', '33', '32s', '32o', '22')
EQ1 = (0.848, 0.660, 0.662, 0.673, 0.634, 0.642, 0.650, 0.639, 0.624, 0.632, 0.618, 0.607, 0.594, 0.627, 0.591, 0.590, 0.586, 0.609, 0.572, 0.587, 0.568, 0.575, 0.567, 0.571, 0.533, 0.832, 0.631, 0.619, 0.625, 0.598, 0.622, 0.603, 0.592, 0.583, 0.583, 0.571, 0.582, 0.541, 0.572, 0.539, 0.565, 0.548, 0.551, 0.529, 0.551, 0.515, 0.526, 0.496, 0.805, 0.599, 0.574, 0.599, 0.569, 0.565, 0.532, 0.549, 0.531, 0.531, 0.514, 0.538, 0.505, 0.513, 0.489, 0.507, 0.483, 0.501, 0.482, 0.499, 0.465, 0.789, 0.570, 0.547, 0.555, 0.530, 0.540, 0.504, 0.517, 0.491, 0.509, 0.478, 0.489, 0.478, 0.481, 0.464, 0.496, 0.451, 0.464, 0.460, 0.746, 0.539, 0.513, 0.537, 0.492, 0.505, 0.478, 0.471, 0.451, 0.462, 0.451, 0.476, 0.433, 0.451, 0.405, 0.446, 0.424, 0.720, 0.496, 0.468, 0.481, 0.472, 0.462, 0.432, 0.469, 0.427, 0.455, 0.402, 0.446, 0.390, 0.443, 0.383, 0.702, 0.481, 0.443, 0.461, 0.427, 0.445, 0.415, 0.426, 0.394, 0.414, 0.367, 0.397, 0.368, 0.660, 0.446, 0.425, 0.445, 0.409, 0.409, 0.396, 0.406, 0.366, 0.392, 0.338, 0.630, 0.425, 0.410, 0.399, 0.379, 0.386, 0.366, 0.381, 0.350, 0.600, 0.414, 0.370, 0.411, 0.366, 0.399, 0.337, 0.567, 0.383, 0.356, 0.376, 0.308, 0.527, 0.353, 0.321, 0.492)
EQ2 = (0.725, 0.497, 0.468, 0.502, 0.461, 0.489, 0.453, 0.440, 0.456, 0.456, 0.434, 0.429, 0.386, 0.452, 0.411, 0.412, 0.391, 0.406, 0.387, 0.406, 0.362, 0.393, 0.370, 0.384, 0.353, 0.689, 0.480, 0.435, 0.462, 0.451, 0.438, 0.414, 0.424, 0.399, 0.413, 0.361, 0.389, 0.360, 0.380, 0.369, 0.358, 0.342, 0.372, 0.334, 0.356, 0.323, 0.359, 0.317, 0.632, 0.457, 0.410, 0.437, 0.410, 0.421, 0.381, 0.381, 0.336, 0.374, 0.333, 0.376, 0.342, 0.298, 0.314, 0.332, 0.295, 0.326, 0.288, 0.342, 0.304, 0.615, 0.464, 0.406, 0.393, 0.359, 0.385, 0.347, 0.361, 0.320, 0.329, 0.312, 0.306, 0.288, 0.302, 0.291, 0.339, 0.268, 0.301, 0.268, 0.586, 0.403, 0.339, 0.364, 0.333, 0.347, 0.323, 0.332, 0.293, 0.294, 0.271, 0.306, 0.262, 0.288, 0.247, 0.289, 0.245, 0.530, 0.363, 0.331, 0.349, 0.298, 0.317, 0.288, 0.278, 0.259, 0.292, 0.223, 0.282, 0.243, 0.271, 0.231, 0.497, 0.357, 0.298, 0.306, 0.284, 0.293, 0.276, 0.281, 0.258, 0.265, 0.221, 0.261, 0.217, 0.483, 0.332, 0.293, 0.292, 0.273, 0.281, 0.233, 0.245, 0.239, 0.241, 0.186, 0.435, 0.306, 0.259, 0.269, 0.247, 0.262, 0.222, 0.255, 0.199, 0.389, 0.264, 0.242, 0.261, 0.205, 0.272, 0.205, 0.354, 0.270, 0.236, 0.248, 0.221, 0.317, 0.238, 0.208, 0.305)
PCT = (0.002, 0.050, 0.039, 0.033, 0.071, 0.062, 0.056, 0.065, 0.097, 0.077, 0.118, 0.127, 0.161, 0.087, 0.173, 0.179, 0.189, 0.124, 0.225, 0.183, 0.261, 0.210, 0.270, 0.234, 0.350, 0.007, 0.080, 0.109, 0.090, 0.152, 0.103, 0.133, 0.167, 0.198, 0.204, 0.240, 0.207, 0.320, 0.231, 0.332, 0.281, 0.302, 0.293, 0.389, 0.290, 0.409, 0.400, 0.475, 0.011, 0.143, 0.216, 0.146, 0.252, 0.284, 0.359, 0.296, 0.368, 0.374, 0.418, 0.341, 0.445, 0.424, 0.513, 0.439, 0.522, 0.463, 0.531, 0.466, 0.603, 0.016, 0.246, 0.311, 0.287, 0.380, 0.326, 0.457, 0.403, 0.501, 0.436, 0.558, 0.507, 0.549, 0.540, 0.615, 0.481, 0.655, 0.609, 0.633, 0.020, 0.338, 0.430, 0.344, 0.487, 0.451, 0.567, 0.585, 0.646, 0.621, 0.664, 0.573, 0.703, 0.670, 0.805, 0.679, 0.754, 0.025, 0.469, 0.594, 0.543, 0.579, 0.624, 0.712, 0.588, 0.730, 0.640, 0.814, 0.673, 0.854, 0.688, 0.869, 0.029, 0.537, 0.694, 0.627, 0.721, 0.685, 0.763, 0.736, 0.842, 0.772, 0.911, 0.827, 0.902, 0.046, 0.676, 0.745, 0.682, 0.793, 0.787, 0.833, 0.799, 0.929, 0.848, 0.968, 0.084, 0.739, 0.781, 0.821, 0.881, 0.860, 0.920, 0.875, 0.959, 0.140, 0.769, 0.893, 0.775, 0.938, 0.824, 0.977, 0.277, 0.863, 0.947, 0.887, 0.995, 0.396, 0.953, 0.986, 0.494)
RANGE_CUTS = (0.08, 0.15, 0.25, 0.4, 0.6)
EQR = (
    (0.848, 0.848, 0.870, 0.863, 0.854),
    (0.557, 0.619, 0.662, 0.673, 0.700),
    (0.553, 0.594, 0.661, 0.641, 0.673),
    (0.469, 0.563, 0.632, 0.642, 0.660),
    (0.463, 0.550, 0.596, 0.648, 0.657),
    (0.411, 0.526, 0.593, 0.644, 0.665),
    (0.364, 0.464, 0.561, 0.620, 0.610),
    (0.350, 0.488, 0.578, 0.564, 0.637),
    (0.328, 0.450, 0.546, 0.586, 0.575),
    (0.335, 0.431, 0.516, 0.585, 0.619),
    (0.302, 0.388, 0.497, 0.548, 0.555),
    (0.326, 0.389, 0.505, 0.552, 0.585),
    (0.290, 0.381, 0.455, 0.522, 0.564),
    (0.310, 0.379, 0.480, 0.551, 0.551),
    (0.273, 0.382, 0.423, 0.499, 0.529),
    (0.305, 0.391, 0.446, 0.525, 0.562),
    (0.267, 0.363, 0.422, 0.472, 0.511),
    (0.321, 0.387, 0.454, 0.520, 0.525),
    (0.285, 0.378, 0.409, 0.497, 0.506),
    (0.353, 0.390, 0.422, 0.493, 0.556),
    (0.294, 0.352, 0.412, 0.502, 0.535),
    (0.300, 0.379, 0.441, 0.468, 0.527),
    (0.275, 0.352, 0.402, 0.471, 0.517),
    (0.310, 0.392, 0.434, 0.495, 0.528),
    (0.273, 0.329, 0.395, 0.455, 0.503),
    (0.686, 0.748, 0.762, 0.800, 0.817),
    (0.365, 0.451, 0.518, 0.560, 0.604),
    (0.315, 0.443, 0.465, 0.565, 0.608),
    (0.331, 0.422, 0.483, 0.549, 0.586),
    (0.332, 0.381, 0.466, 0.539, 0.575),
    (0.344, 0.377, 0.433, 0.532, 0.593),
    (0.319, 0.351, 0.430, 0.505, 0.562),
    (0.336, 0.365, 0.407, 0.503, 0.545),
    (0.315, 0.314, 0.374, 0.463, 0.543),
    (0.319, 0.353, 0.355, 0.438, 0.524),
    (0.285, 0.300, 0.359, 0.441, 0.494),
    (0.323, 0.338, 0.358, 0.454, 0.511),
    (0.310, 0.314, 0.341, 0.410, 0.490),
    (0.325, 0.322, 0.372, 0.433, 0.490),
    (0.285, 0.322, 0.325, 0.411, 0.466),
    (0.323, 0.361, 0.362, 0.404, 0.480),
    (0.288, 0.296, 0.304, 0.381, 0.446),
    (0.310, 0.361, 0.348, 0.398, 0.494),
    (0.289, 0.290, 0.326, 0.388, 0.433),
    (0.297, 0.349, 0.346, 0.425, 0.468),
    (0.255, 0.276, 0.320, 0.385, 0.426),
    (0.330, 0.353, 0.347, 0.403, 0.466),
    (0.250, 0.319, 0.306, 0.380, 0.435),
    (0.643, 0.670, 0.689, 0.712, 0.777),
    (0.335, 0.402, 0.398, 0.494, 0.567),
    (0.302, 0.360, 0.387, 0.457, 0.505),
    (0.368, 0.378, 0.372, 0.477, 0.531),
    (0.312, 0.341, 0.384, 0.444, 0.486),
    (0.309, 0.341, 0.376, 0.425, 0.523),
    (0.328, 0.348, 0.329, 0.398, 0.479),
    (0.338, 0.352, 0.364, 0.416, 0.490),
    (0.277, 0.320, 0.330, 0.353, 0.458),
    (0.316, 0.333, 0.341, 0.412, 0.479),
    (0.261, 0.297, 0.333, 0.341, 0.423),
    (0.298, 0.321, 0.357, 0.387, 0.456),
    (0.269, 0.299, 0.313, 0.352, 0.419),
    (0.300, 0.317, 0.352, 0.388, 0.455),
    (0.248, 0.298, 0.310, 0.338, 0.421),
    (0.281, 0.307, 0.352, 0.358, 0.416),
    (0.273, 0.279, 0.318, 0.366, 0.385),
    (0.309, 0.328, 0.347, 0.367, 0.437),
    (0.258, 0.261, 0.294, 0.324, 0.388),
    (0.292, 0.302, 0.316, 0.380, 0.429),
    (0.242, 0.289, 0.313, 0.347, 0.389),
    (0.565, 0.638, 0.660, 0.682, 0.733),
    (0.363, 0.397, 0.402, 0.460, 0.515),
    (0.307, 0.319, 0.357, 0.420, 0.458),
    (0.336, 0.387, 0.365, 0.406, 0.469),
    (0.278, 0.321, 0.367, 0.383, 0.439),
    (0.335, 0.364, 0.357, 0.381, 0.438),
    (0.300, 0.287, 0.342, 0.366, 0.416),
    (0.326, 0.357, 0.367, 0.371, 0.422),
    (0.252, 0.297, 0.330, 0.332, 0.394),
    (0.317, 0.320, 0.341, 0.346, 0.402),
    (0.257, 0.302, 0.321, 0.330, 0.369),
    (0.296, 0.342, 0.339, 0.343, 0.391),
    (0.278, 0.291, 0.283, 0.318, 0.376),
    (0.283, 0.319, 0.344, 0.360, 0.409),
    (0.248, 0.288, 0.289, 0.315, 0.353),
    (0.281, 0.300, 0.319, 0.339, 0.387),
    (0.226, 0.266, 0.314, 0.332, 0.365),
    (0.255, 0.300, 0.323, 0.361, 0.373),
    (0.243, 0.263, 0.289, 0.302, 0.367),
    (0.529, 0.573, 0.625, 0.646, 0.716),
    (0.329, 0.363, 0.392, 0.429, 0.457),
    (0.310, 0.320, 0.381, 0.369, 0.414),
    (0.314, 0.347, 0.374, 0.394, 0.439),
    (0.295, 0.326, 0.340, 0.339, 0.380),
    (0.337, 0.331, 0.381, 0.355, 0.403),
    (0.264, 0.310, 0.338, 0.337, 0.401),
    (0.285, 0.362, 0.370, 0.366, 0.373),
    (0.280, 0.280, 0.303, 0.338, 0.345),
    (0.279, 0.307, 0.328, 0.355, 0.373),
    (0.240, 0.270, 0.291, 0.307, 0.325),
    (0.262, 0.298, 0.340, 0.340, 0.377),
    (0.259, 0.281, 0.284, 0.288, 0.343),
    (0.280, 0.310, 0.341, 0.348, 0.365),
    (0.274, 0.285, 0.302, 0.315, 0.314),
    (0.273, 0.292, 0.330, 0.350, 0.378),
    (0.256, 0.268, 0.313, 0.328, 0.331),
    (0.449, 0.549, 0.577, 0.628, 0.645),
    (0.327, 0.350, 0.361, 0.420, 0.443),
    (0.302, 0.327, 0.340, 0.352, 0.389),
    (0.307, 0.343, 0.383, 0.406, 0.405),
    (0.267, 0.311, 0.335, 0.378, 0.353),
    (0.308, 0.380, 0.360, 0.368, 0.381),
    (0.278, 0.329, 0.294, 0.318, 0.353),
    (0.282, 0.328, 0.348, 0.348, 0.380),
    (0.257, 0.280, 0.323, 0.321, 0.318),
    (0.284, 0.308, 0.338, 0.347, 0.368),
    (0.235, 0.264, 0.308, 0.309, 0.335),
    (0.259, 0.309, 0.345, 0.347, 0.347),
    (0.236, 0.264, 0.291, 0.277, 0.320),
    (0.294, 0.311, 0.320, 0.337, 0.350),
    (0.235, 0.255, 0.281, 0.310, 0.291),
    (0.438, 0.503, 0.563, 0.601, 0.627),
    (0.327, 0.347, 0.360, 0.389, 0.383),
    (0.262, 0.332, 0.352, 0.350, 0.354),
    (0.297, 0.335, 0.371, 0.352, 0.385),
    (0.275, 0.310, 0.315, 0.339, 0.366),
    (0.306, 0.325, 0.352, 0.382, 0.358),
    (0.256, 0.335, 0.315, 0.330, 0.332),
    (0.289, 0.312, 0.364, 0.355, 0.378),
    (0.255, 0.278, 0.294, 0.322, 0.314),
    (0.246, 0.284, 0.325, 0.341, 0.342),
    (0.236, 0.245, 0.285, 0.307, 0.308),
    (0.280, 0.312, 0.336, 0.340, 0.376),
    (0.227, 0.282, 0.275, 0.293, 0.322),
    (0.393, 0.470, 0.549, 0.561, 0.610),
    (0.320, 0.342, 0.338, 0.397, 0.381),
    (0.293, 0.325, 0.329, 0.351, 0.365),
    (0.282, 0.338, 0.350, 0.380, 0.368),
    (0.280, 0.283, 0.290, 0.340, 0.339),
    (0.280, 0.318, 0.325, 0.359, 0.357),
    (0.249, 0.287, 0.291, 0.322, 0.346),
    (0.286, 0.295, 0.338, 0.353, 0.345),
    (0.257, 0.298, 0.307, 0.290, 0.305),
    (0.258, 0.293, 0.302, 0.322, 0.335),
    (0.224, 0.262, 0.310, 0.287, 0.313),
    (0.370, 0.442, 0.502, 0.559, 0.560),
    (0.304, 0.349, 0.347, 0.404, 0.385),
    (0.286, 0.312, 0.349, 0.357, 0.340),
    (0.293, 0.335, 0.351, 0.352, 0.355),
    (0.265, 0.298, 0.321, 0.344, 0.377),
    (0.266, 0.310, 0.333, 0.346, 0.359),
    (0.249, 0.288, 0.278, 0.307, 0.318),
    (0.287, 0.315, 0.312, 0.350, 0.329),
    (0.221, 0.278, 0.269, 0.306, 0.307),
    (0.402, 0.426, 0.478, 0.541, 0.585),
    (0.314, 0.343, 0.387, 0.361, 0.397),
    (0.283, 0.295, 0.289, 0.360, 0.345),
    (0.314, 0.317, 0.368, 0.365, 0.390),
    (0.278, 0.302, 0.309, 0.343, 0.340),
    (0.286, 0.329, 0.340, 0.345, 0.333),
    (0.254, 0.270, 0.314, 0.293, 0.317),
    (0.358, 0.425, 0.459, 0.507, 0.552),
    (0.305, 0.325, 0.365, 0.343, 0.350),
    (0.271, 0.293, 0.300, 0.325, 0.325),
    (0.298, 0.329, 0.346, 0.338, 0.352),
    (0.261, 0.268, 0.308, 0.300, 0.318),
    (0.404, 0.408, 0.470, 0.499, 0.503),
    (0.281, 0.295, 0.345, 0.336, 0.389),
    (0.254, 0.278, 0.302, 0.278, 0.302),
    (0.363, 0.396, 0.470, 0.487, 0.473),
)
# == END TABLES ==

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}

# hand class index from (hi, lo, suited) in the generator's order: i from A down, j from i down; pair / suited / offsuit
_CLS = {}
_k = 0
for _i in range(12, -1, -1):
    for _j in range(_i, -1, -1):
        if _i == _j:
            _CLS[(_i, _j, False)] = _k
            _k += 1
        else:
            _CLS[(_i, _j, True)] = _k
            _CLS[(_i, _j, False)] = _k + 1
            _k += 2

# Nash-shaped open-shove width (share of hands) at 10 bb, by number of players still to act behind us.
_SHOVE10 = {7: 0.14, 6: 0.16, 5: 0.19, 4: 0.22, 3: 0.26, 2: 0.36, 1: 0.50, 0: 0.50}
# Open-raise width when deep, by players behind.
_OPEN = {7: 0.12, 6: 0.14, 5: 0.16, 4: 0.19, 3: 0.23, 2: 0.28, 1: 0.36, 0: 0.36}


def _hand_class(hole):
    a, b = _RANK[hole[0][0]], _RANK[hole[1][0]]
    hi, lo = (a, b) if a >= b else (b, a)
    suited = hole[0][1] == hole[1][1] and hi != lo
    return _CLS[(hi, lo, suited)]


def _eq_vs_width(k, width):
    """Equity of class k against a hand drawn from the top `width` share of hands (interpolated table)."""
    row = EQR[k]
    cuts = RANGE_CUTS
    if width <= cuts[0]:
        return row[0]
    for i in range(1, len(cuts)):
        if width <= cuts[i]:
            t = (width - cuts[i - 1]) / (cuts[i] - cuts[i - 1])
            return row[i - 1] + t * (row[i] - row[i - 1])
    # beyond the last cut blend towards the vs-random equity
    t = min(1.0, (width - cuts[-1]) / (1.0 - cuts[-1]))
    return row[-1] + t * (EQ1[k] - row[-1])


def _fallback(state):
    try:
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
    except Exception:
        return {"action": "fold"}


class Bot(Agent):
    def __init__(self, seat, config):
        super().__init__(seat, config)
        self.n = int(config.get("n_seats", 8)) if isinstance(config, dict) else 8
        pays = list(config.get("payouts", [50, 30, 20, 0, 0, 0, 0, 0])) if isinstance(config, dict) else [50, 30, 20]
        self.payouts = [float(x) for x in pays]
        self.paid = sum(1 for x in self.payouts if x > 0)
        try:
            self._icm = make_icm(self.payouts, max_players=max(8, self.n))
        except Exception:
            self._icm = None
        # opponent model, per seat
        n = self.n
        self.hands = [0] * n
        self.vpip = [0] * n
        self.pfr = [0] * n
        self.ftr_opp = [0] * n   # faced a preflop raise: opportunities / folds
        self.ftr_fold = [0] * n
        self.ftb_opp = [0] * n   # faced a postflop bet
        self.ftb_fold = [0] * n
        self.agg = [0] * n       # postflop bets/raises
        self.pf_done = [False] * n
        self._street = "preflop"
        self._pf_raised = False
        self._bet_open = False
        self._alive = [True] * n

    # ------------------------------------------------------------------ safety wrapper
    def act(self, state):
        try:
            a = self._decide(state)
            kind, _amt, err = validate_action(a, state["legal_actions"], state["min_raise_to"], state["max_raise_to"])
            if err is None:
                return a
        except Exception:
            pass
        return _fallback(state)

    # ------------------------------------------------------------------ opponent model (untimed)
    def observe(self, event):
        try:
            t = event.get("type")
            if t == "action":
                s = event["seat"]
                act = event["action"]
                if event.get("street", "preflop") == "preflop":
                    if act in ("call", "raise"):
                        if not self.pf_done[s]:
                            self.vpip[s] += 1
                        if act == "raise":
                            self.pfr[s] += 1
                    if act in ("fold", "call", "raise") and self._pf_raised:
                        self.ftr_opp[s] += 1
                        if act == "fold":
                            self.ftr_fold[s] += 1
                    if act == "raise":
                        self._pf_raised = True
                    if act in ("fold", "call", "raise", "check"):
                        self.pf_done[s] = True
                else:
                    if act in ("fold", "call", "raise") and self._bet_open:
                        self.ftb_opp[s] += 1
                        if act == "fold":
                            self.ftb_fold[s] += 1
                    if act == "raise":
                        self.agg[s] += 1
                        self._bet_open = True
            elif t == "hand_start":
                self._pf_raised = False
                self._bet_open = False
                stacks = event.get("stacks")
                for i in range(self.n):
                    self.pf_done[i] = False
                    alive = bool(stacks[i] > 0) if stacks is not None else True
                    self._alive[i] = alive
                    if alive:
                        self.hands[i] += 1
            elif t == "street":
                self._bet_open = False
        except Exception:
            pass

    def _loose(self, s):
        """Estimated share of hands seat s voluntarily plays (prior 0.22)."""
        return (self.vpip[s] + 0.22 * 10) / (self.hands[s] + 10)

    def _ftr(self, s):
        return (self.ftr_fold[s] + 0.6 * 4) / (self.ftr_opp[s] + 4)

    def _ftb(self, s):
        return (self.ftb_fold[s] + 0.5 * 4) / (self.ftb_opp[s] + 4)

    # ------------------------------------------------------------------ decision
    def _decide(self, state):
        me = self.seat
        legal = state["legal_actions"]
        if state["street"] == "preflop":
            return self._preflop(state, legal)
        return self._postflop(state, legal)

    # ---------------- helpers
    @staticmethod
    def _raise(state, target):
        lo, hi = state["min_raise_to"], state["max_raise_to"]
        t = int(target)
        return {"action": "raise", "amount": lo if t < lo else hi if t > hi else t}

    @staticmethod
    def _shove(state, legal):
        if "raise" in legal:
            return {"action": "raise", "amount": state["max_raise_to"]}
        if "call" in legal:
            return {"action": "call"}
        return {"action": "check"}

    def _behind(self, state):
        """Active seats that act after me this preflop round (acting order starts left of the BB)."""
        n = state["n_seats"]
        active = state["active"]
        start = (state["bb_seat"] + 1) % n
        me = state["seat"]
        order = [(start + i) % n for i in range(n)]
        mine = order.index(me)
        return sum(1 for s in order[mine + 1:] if active[s] and s != me)

    def _risk(self, state):
        """ICM-flavoured risk premium: > 0 near the bubble / in the money when we are not the shortest stack."""
        n_alive = state["n_alive"]
        if n_alive <= 1:
            return 0.0
        stacks = state["start_stacks"]
        mine = stacks[self.seat]
        shorter = sum(1 for i in range(len(stacks)) if 0 < stacks[i] < mine)
        if n_alive <= self.paid + 1:
            return 0.06 if shorter > 0 else 0.02
        if n_alive <= self.paid + 2 and shorter > 0:
            return 0.03
        return 0.0

    # ---------------- preflop
    def _preflop(self, state, legal):
        me = state["seat"]
        bb = state["bb"]
        stacks = state["stacks"]
        bets = state["bets"]
        to_call = state["to_call"]
        my_stack = stacks[me]
        eff = my_stack + bets[me]          # my chips at hand start, minus antes
        eff_bb = eff / bb
        k = _hand_class(state["hole"])
        pct = PCT[k]
        behind = self._behind(state)
        active = state["active"]
        n_active = sum(1 for a in active if a)
        top_bet = max(bets)
        hist = state["history_this_hand"]
        raises = [(s, a, amt) for (s, a, amt) in hist if a == "raise"]
        n_raises = len(raises)
        limpers = sum(1 for (s, a, amt) in hist if a == "call")
        risk = self._risk(state)
        all_in = state["all_in"]
        # who is in the pot and still able to play against me
        opp_in = [i for i in range(len(active)) if active[i] and i != me]
        # ---- a shove in front of us (someone all-in, or the call is most of our stack)
        shover_allin = any(all_in[s] for (s, a, amt) in raises)
        facing_shove = to_call > 0 and (shover_allin or to_call >= 0.5 * my_stack)
        if facing_shove:
            return self._call_shove(state, legal, k, raises, opp_in, risk)
        # ---- unopened pot (only blinds / limpers)
        if n_raises == 0:
            if eff_bb <= 16 or (eff_bb <= 20 and behind <= 2):
                # push/fold territory
                width = _SHOVE10.get(behind, 0.14) * (10.0 / max(eff_bb, 2.0)) ** 0.6
                width *= self._steal_mult(state, behind)
                width *= 1.0 - 2.5 * risk
                if limpers:
                    width *= 0.75
                if eff_bb <= 3:
                    width = max(width, 0.6)
                width = min(width, 0.9)
                if pct <= width:
                    return self._shove(state, legal)
                return _fallback(state)
            # deep: open-raise or fold
            width = _OPEN.get(behind, 0.12) * self._steal_mult(state, behind)
            width *= 1.0 - 1.5 * risk
            if limpers:
                width = min(width, 0.16)
            if pct <= width and "raise" in legal:
                size = (2.5 if eff_bb > 25 else 2.2) * bb + limpers * bb
                return self._raise(state, max(size, top_bet * 2))
            if "check" in legal:
                return {"action": "check"}
            # SB complete with playable hands, cheap
            if "call" in legal and to_call <= bb // 2 and pct <= 0.30 and eff_bb > 25:
                return {"action": "call"}
            return {"action": "fold"}
        # ---- facing a raise (not a shove)
        raiser = raises[-1][0]
        raiser_loose = self._loose(raiser)
        if n_raises == 1:
            # 3-bet or shove with premiums, call cheap with pairs / suited broadways
            prem = 0.045 if raiser_loose < 0.30 else 0.07
            if pct <= prem:
                if eff_bb <= 22 or top_bet * 3 >= 0.4 * eff:
                    return self._shove(state, legal)
                if "raise" in legal:
                    return self._raise(state, top_bet * 3 + limpers * bb)
                return {"action": "call"} if "call" in legal else _fallback(state)
            if eff_bb <= 16:
                # resteal shove over loose openers with decent hands
                w = 0.07 + (0.06 if raiser_loose > 0.35 else 0.0) + (0.04 if behind == 0 else 0.0)
                w *= 1.0 - 2.0 * risk
                if pct <= w:
                    return self._shove(state, legal)
                return _fallback(state)
            cheap = to_call <= 0.10 * my_stack and to_call <= 4 * bb
            if "call" in legal and cheap and (pct <= 0.15 or (pct <= 0.22 and behind <= 1)):
                return {"action": "call"}
            return _fallback(state)
        # facing a 3-bet or more
        if pct <= 0.012:
            return self._shove(state, legal)
        if pct <= 0.03 and "call" in legal and to_call <= 0.15 * my_stack and risk == 0.0:
            return {"action": "call"}
        if pct <= 0.016 and "call" in legal and to_call <= 0.25 * my_stack:
            return {"action": "call"}
        return _fallback(state)

    def _steal_mult(self, state, behind):
        """Widen opens/shoves when the players behind fold to raises a lot (typical hour-built bots)."""
        n = state["n_seats"]
        active = state["active"]
        start = (state["bb_seat"] + 1) % n
        order = [(start + i) % n for i in range(n)]
        mine = order.index(state["seat"])
        m = 1.0
        for s in order[mine + 1:]:
            if active[s] and s != state["seat"]:
                f = self._ftr(s)
                m *= 1.0 + 0.5 * (f - 0.6)     # f = 0.9 -> x1.15 per seat; f = 0.3 -> x0.85
        return max(0.6, min(1.8, m))

    def _call_shove(self, state, legal, k, raises, opp_in, risk):
        me = state["seat"]
        stacks = state["stacks"]
        my_stack = stacks[me]
        to_call = state["to_call"]
        pot = state["pot"]
        bb = state["bb"]
        n_opp = len(opp_in)
        # estimate the shover's range width
        shover = raises[-1][0]
        s_bb = (stacks[shover] + state["bets"][shover]) / bb
        loose = self._loose(shover)
        if len(raises) == 1 and not any(a == "call" for (_s, a, _amt) in state["history_this_hand"]):
            width = _SHOVE10.get(4, 0.22) * (10.0 / max(min(s_bb, 40.0), 2.0)) ** 0.6
        else:
            width = 0.10
        width *= max(0.5, min(2.5, loose / 0.22))
        if self.pfr[shover] >= 3 and self.hands[shover] >= 10:
            width *= max(1.0, min(2.0, (self.pfr[shover] / self.hands[shover]) / 0.12))
        width = max(0.04, min(0.95, width))
        eq = _eq_vs_width(k, width)
        if n_opp >= 2:
            eq *= EQ2[k] / max(EQ1[k], 0.01)   # crude multiway discount
        # pot odds first (fast path): a tiny call with any equity
        if to_call <= 0.12 * (my_stack + to_call) and to_call <= 3 * bb and eq >= 0.25:
            return {"action": "call"} if "call" in legal else _fallback(state)
        # exact ICM: compare fold / call outcomes
        if self._icm is not None and state["n_alive"] <= 9:
            fold_s = list(stacks)
            fold_s[me] = my_stack
            fold_s[shover] = stacks[shover] + pot    # shover usually takes the pot if we fold (approximation)
            win_s = list(stacks)
            win_s[me] = my_stack - to_call + pot + to_call
            lose_s = list(stacks)
            lose_s[me] = my_stack - to_call
            lose_s[shover] = stacks[shover] + pot + to_call
            ev_fold = self._icm(fold_s, me)
            ev_call = eq * self._icm(win_s, me) + (1.0 - eq) * self._icm(lose_s, me)
            margin = 0.002 * sum(self.payouts) / 100.0
            if ev_call > ev_fold + margin:
                return self._shove(state, legal) if to_call < my_stack and n_opp == 1 and "raise" in legal and eq >= 0.60 else ({"action": "call"} if "call" in legal else _fallback(state))
            return _fallback(state)
        # fallback: chip EV with a risk premium
        need = to_call / float(pot + to_call) + risk
        if eq >= need:
            return {"action": "call"} if "call" in legal else _fallback(state)
        return _fallback(state)

    # ---------------- postflop
    def _postflop(self, state, legal):
        me = state["seat"]
        hole = state["hole"]
        board = state["board"]
        stacks = state["stacks"]
        bets = state["bets"]
        my_stack = stacks[me]
        pot = state["pot"]
        to_call = state["to_call"]
        street = state["street"]
        active = state["active"]
        all_in = state["all_in"]
        opp = [i for i in range(len(active)) if active[i] and i != me]
        n_opp = len(opp)
        live_opp = [i for i in opp if not all_in[i]]
        risk = self._risk(state)
        cards = [CARD_INT[c] for c in hole] + [CARD_INT[c] for c in board]
        cat = evaluate(cards) >> 20
        # rank / suit features
        h1, h2 = _RANK[hole[0][0]], _RANK[hole[1][0]]
        bh = [_RANK[c[0]] for c in board]
        bmax = max(bh)
        bcount = {}
        for r in bh:
            bcount[r] = bcount.get(r, 0) + 1
        board_paired = any(v >= 2 for v in bcount.values())
        pocket = h1 == h2
        hit1 = h1 in bcount
        hit2 = h2 in bcount
        # flush draw / made flush
        suits = {}
        for c in hole + list(board):
            suits[c[1]] = suits.get(c[1], 0) + 1
        my_suits = (hole[0][1], hole[1][1])
        fd = False
        for s, cnt in suits.items():
            if cnt == 4 and s in my_suits and street != "river":
                fd = True
        # straight draws from the rank mask (my cards must contribute)
        mask = 0
        for r in bh:
            mask |= 1 << r
        mask |= (1 << h1) | (1 << h2)
        outs = 0
        if street != "river" and cat < 4:
            # count ranks that complete a straight
            for r in range(13):
                if mask >> r & 1:
                    continue
                m2 = mask | (1 << r)
                for lo in range(0, 9):
                    if (m2 >> lo) & 31 == 31:
                        outs += 1
                        break
                else:
                    if (m2 & 0x100F) == 0x100F:
                        outs += 1
        oesd = outs >= 2
        gut = outs == 1
        # tiers
        tier = 0
        if cat >= 4:
            tier = 3
        elif cat == 3:
            tier = 3 if (pocket or hit1 or hit2) else 1     # set/trips vs board trips
        elif cat == 2:
            if (hit1 and hit2) or (pocket and board_paired and h1 > min(bcount)):
                tier = 3 if not board_paired or pocket else 2
            elif hit1 or hit2:
                tier = 2 if max(h1, h2) >= bmax else 1    # one pair + board pair
            else:
                tier = 1 if max(h1, h2) >= 10 else 0      # board two pair, kicker play
        elif cat == 1:
            if pocket:
                tier = 2 if h1 > bmax else 1
            elif hit1 or hit2:
                paired = h1 if hit1 else h2
                kicker = h2 if hit1 else h1
                if paired == bmax:
                    tier = 2 if kicker >= 9 or street == "flop" else 1
                else:
                    tier = 1
            else:
                tier = 0       # board pair only
        else:
            tier = 0
        # draws upgrade
        draw_eq = 0.0
        if fd and oesd:
            draw_eq = 0.50 if street == "flop" else 0.33
        elif fd:
            draw_eq = 0.36 if street == "flop" else 0.19
        elif oesd:
            draw_eq = 0.32 if street == "flop" else 0.17
        elif gut:
            draw_eq = 0.17 if street == "flop" else 0.09
        if max(h1, h2) == 12 and not (hit1 or hit2) and street == "flop":
            draw_eq += 0.06    # overcard ace
        # big-stack-vs-us multiway caution
        was_aggressor = self._last_aggressor(state) == me
        # --------- nobody has bet
        if to_call == 0:
            can_bet = "raise" in legal
            if not can_bet:
                return {"action": "check"}
            if tier == 3:
                return self._bet(state, pot, 0.65)
            if tier == 2:
                if street == "river" and n_opp > 1:
                    return {"action": "check"}
                return self._bet(state, pot, 0.55)
            if draw_eq >= 0.30 and street != "river" and (was_aggressor or n_opp == 1):
                return self._bet(state, pot, 0.5)
            if tier == 0 and street == "flop" and was_aggressor and n_opp <= 2 and risk < 0.06:
                # one continuation bet; skip against stations
                station = any(self._ftb(s) < 0.35 for s in live_opp)
                if not station:
                    return self._bet(state, pot, 0.5)
            if tier == 1 and street == "flop" and was_aggressor and n_opp == 1:
                return self._bet(state, pot, 0.45)
            return {"action": "check"}
        # --------- facing a bet
        pot_after = pot + to_call
        need = to_call / float(pot_after)
        bettor = self._last_aggressor(state)
        maniac = bettor is not None and bettor != me and self.hands[bettor] >= 8 and self.agg[bettor] >= 0.6 * self.hands[bettor]
        big = to_call >= 0.5 * my_stack
        if tier == 3:
            if "raise" in legal and my_stack > to_call:
                if cat >= 5 or street == "river" or my_stack <= 3 * pot_after:
                    return self._raise(state, max(state["min_raise_to"], max(bets) * 3))
                return self._raise(state, max(bets) * 3)
            return {"action": "call"}
        if tier == 2:
            if big and not maniac and risk >= 0.03 and n_opp >= 1:
                return {"action": "fold"}
            if need <= 0.40 + (0.10 if maniac else 0.0) or (big and my_stack <= to_call) or (n_opp == 1 and need <= 0.5 and street != "river"):
                return {"action": "call"}
            return {"action": "fold"}
        # draws and weak pairs: pot odds, with implied odds on the flop
        eq = draw_eq
        if tier == 1:
            eq = max(eq, 0.30 if street != "river" else 0.0)
        if street == "river" and tier == 1 and need <= 0.25 and n_opp == 1:
            return {"action": "call"}
        implied = 0.05 if street == "flop" and not big else 0.0
        if eq + implied >= need + risk and not (big and tier < 2 and draw_eq < 0.45):
            return {"action": "call"}
        return {"action": "fold"}

    def _bet(self, state, pot, frac):
        me = state["seat"]
        stack = state["stacks"][me]
        amt = max(state["bb"], int(pot * frac))
        if amt >= 0.7 * stack:
            amt = state["max_raise_to"]
        return self._raise(state, amt)

    @staticmethod
    def _last_aggressor(state):
        hist = state["history_this_hand"]
        last = None
        for (s, a, amt) in hist:
            if a == "raise":
                last = s
        return last


AGENT = Bot
