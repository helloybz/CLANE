import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data') if os.name == 'posix' else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'storage', 'data')

PICKLE_PATH = os.path.join(DATA_PATH, 'pickles')
PAINTER_LIST_URL = ['https://en.wikipedia.org/wiki/List_of_painters_by_name_beginning_with_%22' + chr(i) + '%22' for i
                    in range(ord('A'), ord('Z') + 1)]

URL_STOP_WORDS = ['Help', 'File', 'Wikipedia', 'Special', 'Talk', 'Category', 'Template', 'Portal', 'ISO',
                  'List_of_']

MINIMUM_IMG_NUMBER = 3

#                  ]

ART_MOVEMENTS_KEYWORDS = ['ASCII art', 'Abstract art', 'Outsider Art', 'Abstract expressionism', 'Abstract illusionism',
                          'Academic art', 'Action painting', 'Aestheticism', 'Altermodern', 'American Barbizon school',
                          'American impressionism', 'American realism', 'American Scene Painting', 'Analytical art',
                          'Antipodeans Group', 'Arabesque (European art)', 'Arbeitsrat für Kunst', 'Art & Language',
                          'Art Deco', 'Art Informel', 'Art Nouveau', 'Art photography', 'Arte Povera',
                          'Arts and Crafts movement', 'Ashcan School', 'Assemblage (art)', 'Les Automatistes',
                          'Auto-destructive art', 'Barbizon school', 'Baroque', 'Bauhaus', 'Black Arts Movement',
                          'Classical realism', 'Cloisonnism', 'COBRA (avant-garde movement)', 'Color Field',
                          'Context art', 'Computer art', 'Concrete art', 'Conceptual art', 'Constructivism (art)',
                          'Cubism', 'Cynical realism', 'Dada', 'Danube school', 'Dau-al-Set', 'De Stijl',
                          'Deconstructivism', 'Digital art', 'Environmental art', 'Excessivism', 'Expressionism',
                          'Vienna School of Fantastic Realism', 'Fauvism', 'Feminist art', 'Figurative art',
                          'Figuration Libre', 'Folk art', 'Fluxus', 'Funk art', 'Futurism (art)',
                          'Geometric abstract art', 'Street Art', 'Gutai group', 'Gothic art', 'Happening',
                          'Harlem Renaissance', 'Heidelberg School', 'Hudson River School', 'Hurufiyya movement',
                          'Hypermodernism (art)', 'Hyperrealism (painting)', 'Impressionism', 'Institutional critique',
                          'International Gothic', 'International Typographic Style', 'Kinetic art', 'Kitsch movement',
                          'Land art', 'Les Nabis', 'Letterism', 'Light and Space', 'Lowbrow (art movement)',
                          'Paul Hartal', 'Lyrical abstraction', 'Magic realism', 'Mannerism', 'Massurrealism',
                          'Maximalism', 'Metaphysical painting', 'Mingei', 'Minimalism', 'Modernism',
                          'Modular constructivism', 'Naive art', 'Neoclassicism', 'Neo-Dada', 'Neo-expressionism',
                          'Neo-figurative', 'Neoism', 'Neo-primitivism', 'Net art', 'New Objectivity', 'New Sculpture',
                          'Northwest School (art)', 'Nuclear art', 'Objective abstraction', 'Op Art', 'Orphism (art)',
                          'Photorealism', 'Panfuturism', 'Pixel art', 'Plasticien', 'Plein Air', 'Pointillism',
                          'Pop art', 'Post-impressionism', 'Postminimalism', 'Precisionism', 'Pre-Raphaelitism',
                          'Primitivism', 'Process art', 'Psychedelic art', 'Purism (arts)', 'Qajar art', 'Rasquache',
                          'Rayonism', 'Realism (arts)', 'Regionalism (art)', 'Remodernism', 'Renaissance', 'Rococo',
                          'Romanesque art', 'Romanticism', 'Samikshavad', 'Serial art', 'Shin hanga', 'Shock art',
                          'Sōsaku hanga', 'Socialist realism', 'Sots art', 'Space art', 'Street art', 'Stuckism',
                          'Sumatraism', 'Superflat', 'Suprematism', 'Surrealism', 'Symbolism (arts)', 'Synchromism',
                          'Synthetism', 'Tachisme', 'Toyism', 'Transgressive art', 'Tonalism', 'Ukiyo-e',
                          'Underground comix', 'Vancouver School', 'Vanitas', 'Verdadism', 'Video art',
                          'Viennese Actionism', 'Vorticism']

ART_MOVEMENTS_DICT = {**{item.lower(): item for item in ART_MOVEMENTS_KEYWORDS},
                      **{'mannerist': 'Mannerism',
                         'expressionist': 'Expressionism',
                         'impressionist': 'Impressionism',
                         'post-impressionist': 'Post-impressionism',
                         'conceptual': 'Conceptual art',
                         'romantic': 'Romanticism',
                         'surrealist': 'Surrealism',
                         'realist': 'Realism (arts)',
                         'Realism (arts)': 'Realism (arts)',
                         'academic': 'Academic art',
                         'gothic': 'Gothic art',
                         'neoclassical': 'Neoclassicism',
                         'abstract': 'Abstract art',
                         'fantastic realism': 'Vienna School of Fantastic Realism',
                         }
                      }

WIKIPEDIA_CATEGORIES = ART_MOVEMENTS_KEYWORDS

# WIKIPEDIA_CATEGORIES_KEYS = sum([key for key, value in WIKIPEDIA_CATEGORIES], [])
