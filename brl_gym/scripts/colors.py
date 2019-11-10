

SHADE = 'shade'
STANDARD = 'standard'
EMPHASIS = 'emphasis'

color_library = {}

BLUE = 'blue'
color_library[BLUE] = {}
color_library[BLUE][SHADE] = '#deebf7'
color_library[BLUE][STANDARD] = '#9ecae1'
color_library[BLUE][EMPHASIS]='#3182bd'

# Orange
ORANGE = 'orange'
color_library[ORANGE]={}
color_library[ORANGE][SHADE]='#fee6ce'
color_library[ORANGE][STANDARD]='#fdc086'
color_library[ORANGE][EMPHASIS]='#e6550d'

# Purple
PURPLE = 'purple'
color_library[PURPLE]={}
color_library[PURPLE][SHADE]='#efedf5'
color_library[PURPLE][STANDARD]='#beaed4'
color_library[PURPLE][EMPHASIS]='#756bb1'

# Green
GREEN = 'green'
color_library[GREEN]={}
color_library[GREEN][SHADE]='#e5f5e0'
color_library[GREEN][STANDARD]='#a1d99b'
color_library[GREEN][EMPHASIS]='#31a354'

# GREY
GREY = 'grey'
color_library[GREY]={}
color_library[GREY][SHADE]="#f0f0f0"
color_library[GREY][STANDARD]="#bdbdbd"
color_library[GREY][EMPHASIS]="#636363"

# RED
RED = 'red'
color_library[RED]={}
color_library[RED][SHADE]="#fee0d2"
color_library[RED][STANDARD]="#fc9272"
color_library[RED][EMPHASIS]="#de2d26"

# PINK
PINK = 'pink'
color_library[PINK]={}
color_library[PINK][SHADE]="#fde0dd"
color_library[PINK][STANDARD]="#fa9fb5"
color_library[PINK][EMPHASIS]="#c51b8a"


# OLIVE
OLIVE = 'olive'
color_library[OLIVE]={}
color_library[OLIVE][SHADE]="#90AF47"
color_library[OLIVE][STANDARD]="#728B38"
color_library[OLIVE][EMPHASIS]="#546729"


# YELLOW
YELLOW = 'yellow'
color_library[YELLOW]={}
color_library[YELLOW][SHADE]='#FABF06'
color_library[YELLOW][STANDARD]='#C99904'
color_library[YELLOW][EMPHASIS]='#977303'


random_color = [0.6, 0.6, 0.6]
max_color = [0.4, 0.4, 0.4]
expert_color = [0.0, 0.0, 0.0]

# baselines
brpo_color = color_library[RED]
upmle_color = color_library[PURPLE]
bpo_color = color_library[GREEN]
# ent reward

ent_reward_0 = brpo_color
ent_reward_10 = color_library[YELLOW]
ent_reward_100 = color_library[ORANGE]

# ent feature
ensemble_color = color_library[BLUE]
none_color = color_library[OLIVE]