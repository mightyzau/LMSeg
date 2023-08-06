import numpy as np


if __name__ == '__main__':
    
    cityscape_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    cityscape_classes = [c.split(',')[0] for c in cityscape_classes]
    assert len(cityscape_classes) == 19


    mapillary_classes = ['Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle']
    mapillary_classes = [c.split(',')[0] for c in mapillary_classes]
    assert len(mapillary_classes) == 65

    coco_stuff_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']
    coco_stuff_classes = [c.split(',')[0] for c in coco_stuff_classes]
    assert len(coco_stuff_classes) == 171

    ade20k_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route', 'bed', 'window ', 'grass', 'cabinet', 'sidewalk, pavement', 'person', 'earth, ground', 'door', 'table', 'mountain, mount', 'plant', 'curtain', 'chair', 'car', 'water', 'painting, picture', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock, stone', 'wardrobe, closet, press', 'lamp', 'tub', 'rail', 'cushion', 'base, pedestal, stand', 'box', 'column, pillar', 'signboard, sign', 'chest of drawers, chest, bureau, dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator, icebox', 'grandstand, covered stand', 'path', 'stairs', 'runway', 'case, display case, showcase, vitrine', 'pool table, billiard table, snooker table', 'pillow', 'screen door, screen', 'stairway, staircase', 'river', 'bridge, span', 'bookcase', 'blind, screen', 'coffee table', 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm, palm tree', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel, hut, hutch, shack, shanty', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes', 'pole', 'land, ground, soil', 'bannister, banister, balustrade, balusters, handrail', 'escalator, moving staircase, moving stairway', 'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'buffet, counter, sideboard', 'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship', 'fountain', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy', 'washer, automatic washer, washing machine', 'plaything, toy', 'pool', 'stool', 'barrel, cask', 'basket, handbasket', 'falls', 'tent', 'bag', 'minibike, motorbike', 'cradle', 'oven', 'ball', 'food, solid food', 'step, stair', 'tank, storage tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket, cover', 'sculpture', 'hood, exhaust hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag']
    ade20k_classes = [c.split(',')[0] for c in ade20k_classes]
    assert len(ade20k_classes) == 150


    ade20k_full_results = {'mIoU': 0.7244463028138705, 'fwIoU': 30.89036523655017, 'IoU-wall': 68.6663846562297, 'IoU-building, edifice': 78.84656297841047, 'IoU-sky': 94.17370824146599, 'IoU-tree': 0.006294413834270615, 'IoU-road, route': 0.019691670848792343, 'IoU-floor, flooring': 0.0008096532409117328, 'IoU-ceiling': 0.0, 'IoU-bed': 79.48812738318783, 'IoU-sidewalk, pavement': 0.010719515208497711, 'IoU-earth, ground': 5.22245079538061, 'IoU-cabinet': 51.81257181568243, 'IoU-person, individual, someone, somebody, mortal, soul': 0.20302935045862833, 'IoU-grass': 0.05849944191757346, 'IoU-windowpane, window': 0.0, 'IoU-car, auto, automobile, machine, motorcar': 0.3429739505300444, 'IoU-mountain, mount': 0.0, 'IoU-plant, flora, plant life': 0.3453577481402914, 'IoU-table': 0.018461911173502828, 'IoU-chair': 0.07204257378395125, 'IoU-curtain, drape, drapery, mantle, pall': 0.1455414325325743, 'IoU-door': 0.020870816840590143, 'IoU-sofa, couch, lounge': 0.0, 'IoU-sea': 0.0, 'IoU-painting, picture': 0.005156868616295705, 'IoU-water': 0.0, 'IoU-mirror': 0.0, 'IoU-house': 0.005922009503640852, 'IoU-rug, carpet, carpeting': 0.0, 'IoU-shelf': 0.0, 'IoU-armchair': 0.0, 'IoU-fence, fencing': 0.0, 'IoU-field': 0.0, 'IoU-lamp': 0.0, 'IoU-rock, stone': 0.0, 'IoU-seat': 0.0, 'IoU-river': 0.0, 'IoU-desk': 0.1010666596606919, 'IoU-bathtub, bathing tub, bath, tub': 68.60913597697628, 'IoU-railing, rail': 31.791198116561546, 'IoU-signboard, sign': 0.0, 'IoU-cushion': 0.0, 'IoU-path': 0.0, 'IoU-work surface': 0.0, 'IoU-stairs, steps': 0.0015411337667847597, 'IoU-column, pillar': 0.0, 'IoU-sink': 0.0, 'IoU-wardrobe, closet, press': 0.0, 'IoU-snow': 0.0, 'IoU-refrigerator, icebox': 0.0, 'IoU-base, pedestal, stand': 0.0, 'IoU-bridge, span': 0.0, 'IoU-blind, screen': 0.0, 'IoU-runway': 0.0, 'IoU-cliff, drop, drop-off': 0.0, 'IoU-sand': 0.0, 'IoU-fireplace, hearth, open fireplace': 0.0, 'IoU-pillow': 0.0, 'IoU-screen door, screen': 0.0, 'IoU-toilet, can, commode, crapper, pot, potty, stool, throne': 0.017232907003726403, 'IoU-skyscraper': 0.0, 'IoU-grandstand, covered stand': 0.0, 'IoU-box': 0.0, 'IoU-pool table, billiard table, snooker table': 0.0, 'IoU-palm, palm tree': 0.0, 'IoU-double door': 0.0, 'IoU-coffee table, cocktail table': 0.0, 'IoU-counter': 0.003950224137236864, 'IoU-countertop': 0.0, 'IoU-chest of drawers, chest, bureau, dresser': 0.0, 'IoU-kitchen island': 0.0, 'IoU-boat': 0.0, 'IoU-waterfall, falls': 0.0, 'IoU-stove, kitchen stove, range, kitchen range, cooking stove': 0.0, 'IoU-flower': 0.0, 'IoU-bookcase': 0.19209019858724505, 'IoU-controls': 0.0, 'IoU-book': 0.0, 'IoU-stairway, staircase': 0.0, 'IoU-streetlight, street lamp': 0.0, 'IoU-computer, computing machine, computing device, data processor, electronic computer, information processing system': 0.0, 'IoU-bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle': 84.13931043372858, 'IoU-swivel chair': 0.0, 'IoU-light, light source': 49.18146858963694, 'IoU-bench': 0.0, 'IoU-case, display case, showcase, vitrine': 0.0, 'IoU-towel': 0.0, 'IoU-fountain': 0.0, 'IoU-embankment': 0.0, 'IoU-television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box': 0.0, 'IoU-van': 0.0, 'IoU-hill': 0.0, 'IoU-awning, sunshade, sunblind': 0.0, 'IoU-poster, posting, placard, notice, bill, card': 0.0, 'IoU-truck, motortruck': 0.044051136942600266, 'IoU-airplane, aeroplane, plane': 0.0, 'IoU-pole': 0.005859797031280329, 'IoU-tower': 0.0, 'IoU-court': 0.0, 'IoU-ball': 0.0, 'IoU-aircraft carrier, carrier, flattop, attack aircraft carrier': 0.0, 'IoU-buffet, counter, sideboard': 0.0, 'IoU-hovel, hut, hutch, shack, shanty': 0.0, 'IoU-apparel, wearing apparel, dress, clothes': 0.0, 'IoU-minibike, motorbike': 0.0, 'IoU-animal, animate being, beast, brute, creature, fauna': 0.0, 'IoU-chandelier, pendant, pendent': 0.0, 'IoU-step, stair': 0.0, 'IoU-booth, cubicle, stall, kiosk': 0.0, 'IoU-bicycle, bike, wheel, cycle': 0.0, 'IoU-doorframe, doorcase': 0.0, 'IoU-sconce': 0.0, 'IoU-pond': 0.0, 'IoU-trade name, brand name, brand, marque': 0.0, 'IoU-bannister, banister, balustrade, balusters, handrail': 0.0, 'IoU-bag': 0.0, 'IoU-traffic light, traffic signal, stoplight': 0.0, 'IoU-gazebo': 0.0, 'IoU-escalator, moving staircase, moving stairway': 0.0, 'IoU-land, ground, soil': 0.0, 'IoU-board, plank': 0.0, 'IoU-arcade machine': 0.0, 'IoU-eiderdown, duvet, continental quilt': 0.0, 'IoU-bar': 0.0, 'IoU-stall, stand, sales booth': 0.0, 'IoU-playground': 0.0, 'IoU-ship': 0.0, 'IoU-ottoman, pouf, pouffe, puff, hassock': 0.0, 'IoU-ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin': 0.0, 'IoU-bottle': 0.0, 'IoU-cradle': 0.0, 'IoU-pot, flowerpot': 0.0, 'IoU-conveyer belt, conveyor belt, conveyer, conveyor, transporter': 0.0, 'IoU-train, railroad train': 0.0, 'IoU-stool': 0.0, 'IoU-lake': 0.0, 'IoU-tank, storage tank': 0.0, 'IoU-ice, water ice': 0.0, 'IoU-basket, handbasket': 0.05393608232876432, 'IoU-manhole': 0.0, 'IoU-tent, collapsible shelter': 0.0, 'IoU-canopy': 0.0, 'IoU-microwave, microwave oven': 0.0, 'IoU-barrel, cask': 0.0, 'IoU-dirt track': 0.0, 'IoU-beam': 0.0, 'IoU-dishwasher, dish washer, dishwashing machine': 0.0, 'IoU-plate': 0.0, 'IoU-screen, crt screen': 0.0, 'IoU-ruins': 0.0, 'IoU-washer, automatic washer, washing machine': 0.0, 'IoU-blanket, cover': 0.0, 'IoU-plaything, toy': 0.0, 'IoU-food, solid food': 0.0, 'IoU-screen, silver screen, projection screen': 0.0, 'IoU-oven': 0.0, 'IoU-stage': 0.0, 'IoU-beacon, lighthouse, beacon light, pharos': 0.0, 'IoU-umbrella': 0.0, 'IoU-sculpture': 0.0, 'IoU-aqueduct': 0.0, 'IoU-container': 0.0, 'IoU-scaffolding, staging': 0.0, 'IoU-hood, exhaust hood': 0.0, 'IoU-curb, curbing, kerb': 0.0, 'IoU-roller coaster': 0.0, 'IoU-horse, equus caballus': 0.0, 'IoU-catwalk': 0.0, 'IoU-glass, drinking glass': 0.0, 'IoU-vase': 0.0, 'IoU-central reservation': 0.0, 'IoU-carousel': 0.0, 'IoU-radiator': 0.0, 'IoU-closet': 0.0, 'IoU-machine': 0.0, 'IoU-pier, wharf, wharfage, dock': 0.0, 'IoU-fan': 0.0, 'IoU-inflatable bounce game': 0.0, 'IoU-pitch': 0.0, 'IoU-paper': 0.0, 'IoU-arcade, colonnade': 0.0, 'IoU-hot tub': 0.0, 'IoU-helicopter': 0.0, 'IoU-tray': 0.0, 'IoU-partition, divider': 0.0, 'IoU-vineyard': 0.0, 'IoU-bowl': 0.0, 'IoU-bullring': 0.0, 'IoU-flag': 0.0, 'IoU-pot': 0.0, 'IoU-footbridge, overcrossing, pedestrian bridge': 0.0, 'IoU-shower': 0.0, 'IoU-bag, traveling bag, travelling bag, grip, suitcase': 0.0, 'IoU-bulletin board, notice board': 0.0, 'IoU-confessional booth': 0.0, 'IoU-trunk, tree trunk, bole': 0.0, 'IoU-forest': 0.0, 'IoU-elevator door': 0.0, 'IoU-laptop, laptop computer': 0.0, 'IoU-instrument panel': 0.0, 'IoU-bucket, pail': 0.0, 'IoU-tapestry, tapis': 0.0, 'IoU-platform': 0.0, 'IoU-jacket': 0.0, 'IoU-gate': 0.0, 'IoU-monitor, monitoring device': 0.0, 'IoU-telephone booth, phone booth, call box, telephone box, telephone kiosk': 0.0, 'IoU-spotlight, spot': 0.0, 'IoU-ring': 0.0, 'IoU-control panel': 0.0, 'IoU-blackboard, chalkboard': 0.0, 'IoU-air conditioner, air conditioning': 0.0, 'IoU-chest': 0.0, 'IoU-clock': 0.0, 'IoU-sand dune': 0.0, 'IoU-pipe, pipage, piping': 0.0, 'IoU-vault': 0.0, 'IoU-table football': 0.0, 'IoU-cannon': 0.0, 'IoU-swimming pool, swimming bath, natatorium': 0.0, 'IoU-fluorescent, fluorescent fixture': 0.0, 'IoU-statue': 0.0, 'IoU-loudspeaker, speaker, speaker unit, loudspeaker system, speaker system': 0.0, 'IoU-exhibitor': 0.0, 'IoU-ladder': 0.0, 'IoU-carport': 0.0, 'IoU-dam': 0.0, 'IoU-pulpit': 0.0, 'IoU-skylight, fanlight': 0.0, 'IoU-water tower': 0.0, 'IoU-grill, grille, grillwork': 0.0, 'IoU-display board': 0.0, 'IoU-pane, pane of glass, window glass': 0.0, 'IoU-rubbish, trash, scrap': 0.0, 'IoU-ice rink': 0.0, 'IoU-fruit': 0.0, 'IoU-patio': 0.0, 'IoU-vending machine': 0.0, 'IoU-telephone, phone, telephone set': 0.0, 'IoU-net': 0.0, 'IoU-backpack, back pack, knapsack, packsack, rucksack, haversack': 0.0, 'IoU-jar': 0.0, 'IoU-track': 0.0, 'IoU-magazine': 0.0, 'IoU-shutter': 0.0, 'IoU-roof': 0.0, 'IoU-banner, streamer': 0.0, 'IoU-landfill': 0.0, 'IoU-post': 0.0, 'IoU-altarpiece, reredos': 0.0, 'IoU-hat, chapeau, lid': 0.0, 'IoU-arch, archway': 0.0, 'IoU-table game': 0.0, 'IoU-bag, handbag, pocketbook, purse': 0.0, 'IoU-document, written document, papers': 0.0, 'IoU-dome': 0.0, 'IoU-pier': 0.0, 'IoU-shanties': 0.0, 'IoU-forecourt': 0.0, 'IoU-crane': 0.0, 'IoU-dog, domestic dog, canis familiaris': 0.0, 'IoU-piano, pianoforte, forte-piano': 0.0, 'IoU-drawing': 0.0, 'IoU-cabin': 0.0, 'IoU-ad, advertisement, advertizement, advertising, advertizing, advert': 0.0, 'IoU-amphitheater, amphitheatre, coliseum': 0.0, 'IoU-monument': 0.0, 'IoU-henhouse': 0.0, 'IoU-cockpit': 0.0, 'IoU-heater, warmer': 0.0, 'IoU-windmill, aerogenerator, wind generator': 0.0, 'IoU-pool': 0.0, 'IoU-elevator, lift': 0.0, 'IoU-decoration, ornament, ornamentation': 0.0, 'IoU-labyrinth': 0.0, 'IoU-text, textual matter': 0.0, 'IoU-printer': 0.0, 'IoU-mezzanine, first balcony': 0.0, 'IoU-mattress': 0.0, 'IoU-straw': 0.0, 'IoU-stalls': 0.0, 'IoU-patio, terrace': 0.0, 'IoU-billboard, hoarding': 0.0, 'IoU-bus stop': 0.0, 'IoU-trouser, pant': 0.0, 'IoU-console table, console': 0.0, 'IoU-rack': 0.0, 'IoU-notebook': 0.0, 'IoU-shrine': 0.0, 'IoU-pantry': 0.0, 'IoU-cart': 0.0, 'IoU-steam shovel': 0.0, 'IoU-porch': 0.0, 'IoU-postbox, mailbox, letter box': 0.0, 'IoU-figurine, statuette': 0.0, 'IoU-recycling bin': 0.0, 'IoU-folding screen': 0.0, 'IoU-telescope': 0.0, 'IoU-deck chair, beach chair': 0.0, 'IoU-kennel': 0.0, 'IoU-coffee maker': 0.0, "IoU-altar, communion table, lord's table": 0.0, 'IoU-fish': 0.0, 'IoU-easel': 0.0, 'IoU-artificial golf green': 0.0, 'IoU-iceberg': 0.0, 'IoU-candlestick, candle holder': 0.0, 'IoU-shower stall, shower bath': 0.0, 'IoU-television stand': 0.0, 'IoU-wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle': 0.0, 'IoU-skeleton': 0.0, 'IoU-grand piano, grand': 0.0, 'IoU-candy, confect': 0.0, 'IoU-grille door': 0.0, 'IoU-pedestal, plinth, footstall': 0.0, 'IoU-jersey, t-shirt, tee shirt': 0.0, 'IoU-shoe': 0.0, 'IoU-gravestone, headstone, tombstone': 0.0, 'IoU-shanty': 0.0, 'IoU-structure': 0.0, 'IoU-rocking chair, rocker': 0.0, 'IoU-bird': 0.0, 'IoU-place mat': 0.0, 'IoU-tomb': 0.0, 'IoU-big top': 0.0, 'IoU-gas pump, gasoline pump, petrol pump, island dispenser': 0.0, 'IoU-lockers': 0.0, 'IoU-cage': 0.0, 'IoU-finger': 0.0, 'IoU-bleachers': 0.0, 'IoU-ferris wheel': 0.0, 'IoU-hairdresser chair': 0.0, 'IoU-mat': 0.0, 'IoU-stands': 0.0, 'IoU-aquarium, fish tank, marine museum': 0.0, 'IoU-streetcar, tram, tramcar, trolley, trolley car': 0.0, 'IoU-napkin, table napkin, serviette': 0.0, 'IoU-dummy': 0.0, 'IoU-booklet, brochure, folder, leaflet, pamphlet': 0.0, 'IoU-sand trap': 0.0, 'IoU-shop, store': 0.0, 'IoU-table cloth': 0.0, 'IoU-service station': 0.0, 'IoU-coffin': 0.0, 'IoU-drawer': 0.0, 'IoU-cages': 0.0, 'IoU-slot machine, coin machine': 0.0, 'IoU-balcony': 0.0, 'IoU-volleyball court': 0.0, 'IoU-table tennis': 0.0, 'IoU-control table': 0.0, 'IoU-shirt': 0.0, 'IoU-merchandise, ware, product': 0.0, 'IoU-railway': 0.0, 'IoU-parterre': 0.0, 'IoU-chimney': 0.0, 'IoU-can, tin, tin can': 0.0, 'IoU-tanks': 0.0, 'IoU-fabric, cloth, material, textile': 0.0, 'IoU-alga, algae': 0.0, 'IoU-system': 0.0, 'IoU-map': 0.0, 'IoU-greenhouse': 0.0, 'IoU-mug': 0.0, 'IoU-barbecue': 0.0, 'IoU-trailer': 0.0, 'IoU-toilet tissue, toilet paper, bathroom tissue': 0.0, 'IoU-organ': 0.0, 'IoU-dishrag, dishcloth': 0.0, 'IoU-island': 0.0, 'IoU-keyboard': 0.0, 'IoU-trench': 0.0, 'IoU-basket, basketball hoop, hoop': 0.0, 'IoU-steering wheel, wheel': 0.0, 'IoU-pitcher, ewer': 0.0, 'IoU-goal': 0.0, 'IoU-bread, breadstuff, staff of life': 0.0, 'IoU-beds': 0.0, 'IoU-wood': 0.0, 'IoU-file cabinet': 0.0, 'IoU-newspaper, paper': 0.0, 'IoU-motorboat': 0.0, 'IoU-rope': 0.0, 'IoU-guitar': 0.0, 'IoU-rubble': 0.0, 'IoU-scarf': 0.0, 'IoU-barrels': 0.0, 'IoU-cap': 0.0, 'IoU-leaves': 0.0, 'IoU-control tower': 0.0, 'IoU-dashboard': 0.0, 'IoU-bandstand': 0.0, 'IoU-lectern': 0.0, 'IoU-switch, electric switch, electrical switch': 0.0, 'IoU-baseboard, mopboard, skirting board': 0.0, 'IoU-shower room': 0.0, 'IoU-smoke': 0.0, 'IoU-faucet, spigot': 0.0, 'IoU-bulldozer': 0.0, 'IoU-saucepan': 0.0, 'IoU-shops': 0.0, 'IoU-meter': 0.0, 'IoU-crevasse': 0.0, 'IoU-gear': 0.0, 'IoU-candelabrum, candelabra': 0.0, 'IoU-sofa bed': 0.0, 'IoU-tunnel': 0.0, 'IoU-pallet': 0.0, 'IoU-wire, conducting wire': 0.0, 'IoU-kettle, boiler': 0.0, 'IoU-bidet': 0.0, 'IoU-baby buggy, baby carriage, carriage, perambulator, pram, stroller, go-cart, pushchair, pusher': 0.0, 'IoU-music stand': 0.0, 'IoU-pipe, tube': 0.0, 'IoU-cup': 0.0, 'IoU-parking meter': 0.0, 'IoU-ice hockey rink': 0.0, 'IoU-shelter': 0.0, 'IoU-weeds': 0.0, 'IoU-temple': 0.0, 'IoU-patty, cake': 0.0, 'IoU-ski slope': 0.0, 'IoU-panel': 0.0, 'IoU-wallet': 0.0, 'IoU-wheel': 0.0, 'IoU-towel rack, towel horse': 0.0, 'IoU-roundabout': 0.0, 'IoU-canister, cannister, tin': 0.0, 'IoU-rod': 0.0, 'IoU-soap dispenser': 0.0, 'IoU-bell': 0.0, 'IoU-canvas': 0.0, 'IoU-box office, ticket office, ticket booth': 0.0, 'IoU-teacup': 0.0, 'IoU-trellis': 0.0, 'IoU-workbench': 0.0, 'IoU-valley, vale': 0.0, 'IoU-toaster': 0.0, 'IoU-knife': 0.0, 'IoU-podium': 0.0, 'IoU-ramp': 0.0, 'IoU-tumble dryer': 0.0, 'IoU-fireplug, fire hydrant, plug': 0.0, 'IoU-gym shoe, sneaker, tennis shoe': 0.0, 'IoU-lab bench': 0.0, 'IoU-equipment': 0.0, 'IoU-rocky formation': 0.0, 'IoU-plastic': 0.0, 'IoU-calendar': 0.0, 'IoU-caravan': 0.0, 'IoU-check-in-desk': 0.0, 'IoU-ticket counter': 0.0, 'IoU-brush': 0.0, 'IoU-mill': 0.0, 'IoU-covered bridge': 0.0, 'IoU-bowling alley': 0.0, 'IoU-hanger': 0.0, 'IoU-excavator': 0.0, 'IoU-trestle': 0.0, 'IoU-revolving door': 0.0, 'IoU-blast furnace': 0.0, 'IoU-scale, weighing machine': 0.0, 'IoU-projector': 0.0, 'IoU-soap': 0.0, 'IoU-locker': 0.0, 'IoU-tractor': 0.0, 'IoU-stretcher': 0.0, 'IoU-frame': 0.0, 'IoU-grating': 0.0, 'IoU-alembic': 0.0, 'IoU-candle, taper, wax light': 0.0, 'IoU-barrier': 0.0, 'IoU-cardboard': 0.0, 'IoU-cave': 0.0, 'IoU-puddle': 0.0, 'IoU-tarp': 0.0, 'IoU-price tag': 0.0, 'IoU-watchtower': 0.0, 'IoU-meters': 0.0, 'IoU-light bulb, lightbulb, bulb, incandescent lamp, electric light, electric-light bulb': 0.0, 'IoU-tracks': 0.0, 'IoU-hair dryer': 0.0, 'IoU-skirt': 0.0, 'IoU-viaduct': 0.0, 'IoU-paper towel': 0.0, 'IoU-coat': 0.0, 'IoU-sheet': 0.0, 'IoU-fire extinguisher, extinguisher, asphyxiator': 0.0, 'IoU-water wheel': 0.0, 'IoU-pottery, clayware': 0.0, 'IoU-magazine rack': 0.0, 'IoU-teapot': 0.0, 'IoU-microphone, mike': 0.0, 'IoU-support': 0.0, 'IoU-forklift': 0.0, 'IoU-canyon': 0.0, 'IoU-cash register, register': 0.0, 'IoU-leaf, leafage, foliage': 0.0, 'IoU-remote control, remote': 0.0, 'IoU-soap dish': 0.0, 'IoU-windshield, windscreen': 0.0, 'IoU-cat': 0.0, 'IoU-cue, cue stick, pool cue, pool stick': 0.0, 'IoU-vent, venthole, vent-hole, blowhole': 0.0, 'IoU-videos': 0.0, 'IoU-shovel': 0.0, 'IoU-eaves': 0.0, 'IoU-antenna, aerial, transmitting aerial': 0.0, 'IoU-shipyard': 0.0, 'IoU-hen, biddy': 0.0, 'IoU-traffic cone': 0.0, 'IoU-washing machines': 0.0, 'IoU-truck crane': 0.0, 'IoU-cds': 0.0, 'IoU-niche': 0.0, 'IoU-scoreboard': 0.0, 'IoU-briefcase': 0.0, 'IoU-boot': 0.0, 'IoU-sweater, jumper': 0.0, 'IoU-hay': 0.0, 'IoU-pack': 0.0, 'IoU-bottle rack': 0.0, 'IoU-glacier': 0.0, 'IoU-pergola': 0.0, 'IoU-building materials': 0.0, 'IoU-television camera': 0.0, 'IoU-first floor': 0.0, 'IoU-rifle': 0.0, 'IoU-tennis table': 0.0, 'IoU-stadium': 0.0, 'IoU-safety belt': 0.0, 'IoU-cover': 0.0, 'IoU-dish rack': 0.0, 'IoU-synthesizer': 0.0, 'IoU-pumpkin': 0.0, 'IoU-gutter': 0.0, 'IoU-fruit stand': 0.0, 'IoU-ice floe, floe': 0.0, 'IoU-handle, grip, handgrip, hold': 0.0, 'IoU-wheelchair': 0.0, 'IoU-mousepad, mouse mat': 0.0, 'IoU-diploma': 0.0, 'IoU-fairground ride': 0.0, 'IoU-radio': 0.0, 'IoU-hotplate': 0.0, 'IoU-junk': 0.0, 'IoU-wheelbarrow': 0.0, 'IoU-stream': 0.0, 'IoU-toll plaza': 0.0, 'IoU-punching bag': 0.0, 'IoU-trough': 0.0, 'IoU-throne': 0.0, 'IoU-chair desk': 0.0, 'IoU-weighbridge': 0.0, 'IoU-extractor fan': 0.0, 'IoU-hanging clothes': 0.0, 'IoU-dish, dish aerial, dish antenna, saucer': 0.0, 'IoU-alarm clock, alarm': 0.0, 'IoU-ski lift': 0.0, 'IoU-chain': 0.0, 'IoU-garage': 0.0, 'IoU-mechanical shovel': 0.0, 'IoU-wine rack': 0.0, 'IoU-tramway': 0.0, 'IoU-treadmill': 0.0, 'IoU-menu': 0.0, 'IoU-block': 0.0, 'IoU-well': 0.0, 'IoU-witness stand': 0.0, 'IoU-branch': 0.0, 'IoU-duck': 0.0, 'IoU-casserole': 0.0, 'IoU-frying pan': 0.0, 'IoU-desk organizer': 0.0, 'IoU-mast': 0.0, 'IoU-spectacles, specs, eyeglasses, glasses': 0.0, 'IoU-service elevator': 0.0, 'IoU-dollhouse': 0.0, 'IoU-hammock': 0.0, 'IoU-clothes hanging': 0.0, 'IoU-photocopier': 0.0, 'IoU-notepad': 0.0, 'IoU-golf cart': 0.0, 'IoU-footpath': 0.0, 'IoU-cross': 0.0, 'IoU-baptismal font': 0.0, 'IoU-boiler': 0.0, 'IoU-skip': 0.0, 'IoU-rotisserie': 0.0, 'IoU-tables': 0.0, 'IoU-water mill': 0.0, 'IoU-helmet': 0.0, 'IoU-cover curtain': 0.0, 'IoU-brick': 0.0, 'IoU-table runner': 0.0, 'IoU-ashtray': 0.0, 'IoU-street box': 0.0, 'IoU-stick': 0.0, 'IoU-hangers': 0.0, 'IoU-cells': 0.0, 'IoU-urinal': 0.0, 'IoU-centerpiece': 0.0, 'IoU-portable fridge': 0.0, 'IoU-dvds': 0.0, 'IoU-golf club': 0.0, 'IoU-skirting board': 0.0, 'IoU-water cooler': 0.0, 'IoU-clipboard': 0.0, 'IoU-camera, photographic camera': 0.0, 'IoU-pigeonhole': 0.0, 'IoU-chips': 0.0, 'IoU-food processor': 0.0, 'IoU-post box': 0.0, 'IoU-lid': 0.0, 'IoU-drum': 0.0, 'IoU-blender': 0.0, 'IoU-cave entrance': 0.0, 'IoU-dental chair': 0.0, 'IoU-obelisk': 0.0, 'IoU-canoe': 0.0, 'IoU-mobile': 0.0, 'IoU-monitors': 0.0, 'IoU-pool ball': 0.0, 'IoU-cue rack': 0.0, 'IoU-baggage carts': 0.0, 'IoU-shore': 0.0, 'IoU-fork': 0.0, 'IoU-paper filer': 0.0, 'IoU-bicycle rack': 0.0, 'IoU-coat rack': 0.0, 'IoU-garland': 0.0, 'IoU-sports bag': 0.0, 'IoU-fish tank': 0.0, 'IoU-towel dispenser': 0.0, 'IoU-carriage': 0.0, 'IoU-brochure': 0.0, 'IoU-plaque': 0.0, 'IoU-stringer': 0.0, 'IoU-iron': 0.0, 'IoU-spoon': 0.0, 'IoU-flag pole': 0.0, 'IoU-toilet brush': 0.0, 'IoU-book stand': 0.0, 'IoU-water faucet, water tap, tap, hydrant': 0.0, 'IoU-ticket office': 0.0, 'IoU-broom': 0.0, 'IoU-dvd': 0.0, 'IoU-ice bucket': 0.0, 'IoU-carapace, shell, cuticle, shield': 0.0, 'IoU-tureen': 0.0, 'IoU-folders': 0.0, 'IoU-chess': 0.0, 'IoU-root': 0.0, 'IoU-sewing machine': 0.0, 'IoU-model': 0.0, 'IoU-pen': 0.0, 'IoU-violin': 0.0, 'IoU-sweatshirt': 0.0, 'IoU-recycling materials': 0.0, 'IoU-mitten': 0.0, 'IoU-chopping board, cutting board': 0.0, 'IoU-mask': 0.0, 'IoU-log': 0.0, 'IoU-mouse, computer mouse': 0.0, 'IoU-grill': 0.0, 'IoU-hole': 0.0, 'IoU-target': 0.0, 'IoU-trash bag': 0.0, 'IoU-chalk': 0.0, 'IoU-sticks': 0.0, 'IoU-balloon': 0.0, 'IoU-score': 0.0, 'IoU-hair spray': 0.0, 'IoU-roll': 0.0, 'IoU-runner': 0.0, 'IoU-engine': 0.0, 'IoU-inflatable glove': 0.0, 'IoU-games': 0.0, 'IoU-pallets': 0.0, 'IoU-baskets': 0.0, 'IoU-coop': 0.0, 'IoU-dvd player': 0.0, 'IoU-rocking horse': 0.0, 'IoU-buckets': 0.0, 'IoU-bread rolls': 0.0, 'IoU-shawl': 0.0, 'IoU-watering can': 0.0, 'IoU-spotlights': 0.0, 'IoU-post-it': 0.0, 'IoU-bowls': 0.0, 'IoU-security camera': 0.0, 'IoU-runner cloth': 0.0, 'IoU-lock': 0.0, 'IoU-alarm, warning device, alarm system': 0.0, 'IoU-side': 0.0, 'IoU-roulette': 0.0, 'IoU-bone': 0.0, 'IoU-cutlery': 0.0, 'IoU-pool balls': 0.0, 'IoU-wheels': 0.0, 'IoU-spice rack': 0.0, 'IoU-plant pots': 0.0, 'IoU-towel ring': 0.0, 'IoU-bread box': 0.0, 'IoU-video': 0.0, 'IoU-funfair': 0.0, 'IoU-breads': 0.0, 'IoU-tripod': 0.0, 'IoU-ironing board': 0.0, 'IoU-skimmer': 0.0, 'IoU-hollow': 0.0, 'IoU-scratching post': 0.0, 'IoU-tricycle': 0.0, 'IoU-file box': 0.0, 'IoU-mountain pass': 0.0, 'IoU-tombstones': 0.0, 'IoU-cooker': 0.0, 'IoU-card game, cards': 0.0, 'IoU-golf bag': 0.0, 'IoU-towel paper': 0.0, 'IoU-chaise lounge': 0.0, 'IoU-sun': 0.0, 'IoU-toilet paper holder': 0.0, 'IoU-rake': 0.0, 'IoU-key': 0.0, 'IoU-umbrella stand': 0.0, 'IoU-dartboard': 0.0, 'IoU-transformer': 0.0, 'IoU-fireplace utensils': 0.0, 'IoU-sweatshirts': 0.0, 'IoU-cellular telephone, cellular phone, cellphone, cell, mobile phone': 0.0, 'IoU-tallboy': 0.0, 'IoU-stapler': 0.0, 'IoU-sauna': 0.0, 'IoU-test tube': 0.0, 'IoU-palette': 0.0, 'IoU-shopping carts': 0.0, 'IoU-tools': 0.0, 'IoU-push button, push, button': 0.0, 'IoU-star': 0.0, 'IoU-roof rack': 0.0, 'IoU-barbed wire': 0.0, 'IoU-spray': 0.0, 'IoU-ear': 0.0, 'IoU-sponge': 0.0, 'IoU-racket': 0.0, 'IoU-tins': 0.0, 'IoU-eyeglasses': 0.0, 'IoU-file': 0.0, 'IoU-scarfs': 0.0, 'IoU-sugar bowl': 0.0, 'IoU-flip flop': 0.0, 'IoU-headstones': 0.0, 'IoU-laptop bag': 0.0, 'IoU-leash': 0.0, 'IoU-climbing frame': 0.0, 'IoU-suit hanger': 0.0, 'IoU-floor spotlight': 0.0, 'IoU-plate rack': 0.0, 'IoU-sewer': 0.0, 'IoU-hard drive': 0.0, 'IoU-sprinkler': 0.0, 'IoU-tools box': 0.0, 'IoU-necklace': 0.0, 'IoU-bulbs': 0.0, 'IoU-steel industry': 0.0, 'IoU-club': 0.0, 'IoU-jack': 0.0, 'IoU-door bars': 0.0, 'IoU-control panel, instrument panel, control board, board, panel': 0.0, 'IoU-hairbrush': 0.0, 'IoU-napkin holder': 0.0, 'IoU-office': 0.0, 'IoU-smoke detector': 0.0, 'IoU-utensils': 0.0, 'IoU-apron': 0.0, 'IoU-scissors': 0.0, 'IoU-terminal': 0.0, 'IoU-grinder': 0.0, 'IoU-entry phone': 0.0, 'IoU-newspaper stand': 0.0, 'IoU-pepper shaker': 0.0, 'IoU-onions': 0.0, 'IoU-central processing unit, cpu, c p u , central processor, processor, mainframe': 0.0, 'IoU-tape': 0.0, 'IoU-bat': 0.0, 'IoU-coaster': 0.0, 'IoU-calculator': 0.0, 'IoU-potatoes': 0.0, 'IoU-luggage rack': 0.0, 'IoU-salt': 0.0, 'IoU-street number': 0.0, 'IoU-viewpoint': 0.0, 'IoU-sword': 0.0, 'IoU-cd': 0.0, 'IoU-rowing machine': 0.0, 'IoU-plug': 0.0, 'IoU-andiron, firedog, dog, dog-iron': 0.0, 'IoU-pepper': 0.0, 'IoU-tongs': 0.0, 'IoU-bonfire': 0.0, 'IoU-dog dish': 0.0, 'IoU-belt': 0.0, 'IoU-dumbbells': 0.0, 'IoU-videocassette recorder, vcr': 0.0, 'IoU-hook': 0.0, 'IoU-envelopes': 0.0, 'IoU-shower faucet': 0.0, 'IoU-watch': 0.0, 'IoU-padlock': 0.0, 'IoU-swimming pool ladder': 0.0, 'IoU-spanners': 0.0, 'IoU-gravy boat': 0.0, 'IoU-notice board': 0.0, 'IoU-trash bags': 0.0, 'IoU-fire alarm': 0.0, 'IoU-ladle': 0.0, 'IoU-stethoscope': 0.0, 'IoU-rocket': 0.0, 'IoU-funnel': 0.0, 'IoU-bowling pins': 0.0, 'IoU-valve': 0.0, 'IoU-thermometer': 0.0, 'IoU-cups': 0.0, 'IoU-spice jar': 0.0, 'IoU-night light': 0.0, 'IoU-soaps': 0.0, 'IoU-games table': 0.0, 'IoU-slotted spoon': 0.0, 'IoU-reel': 0.0, 'IoU-scourer': 0.0, 'IoU-sleeping robe': 0.0, 'IoU-desk mat': 0.0, 'IoU-dumbbell': 0.0, 'IoU-hammer': 0.0, 'IoU-tie': 0.0, 'IoU-typewriter': 0.0, 'IoU-shaker': 0.0, 'IoU-cheese dish': 0.0, 'IoU-sea star': 0.0, 'IoU-racquet': 0.0, 'IoU-butane gas cylinder': 0.0, 'IoU-paper weight': 0.0, 'IoU-shaving brush': 0.0, 'IoU-sunglasses': 0.0, 'IoU-gear shift': 0.0, 'IoU-towel rail': 0.0, 'IoU-adding machine, totalizer, totaliser': 0.0, 'mACC': 0.879072079754892, 'pACC': 36.30916304299268, 'ACC-wall': 86.03931658849282, 'ACC-building, edifice': 91.66059746349755}



    import pdb; pdb.set_trace()




    ious_not_in_4dataset = {}
    ious_in_4dataset = {}
    for k in ade20k_full_results:
        if 'IoU-' not in k:
            continue

        category = k[4:].split(',')[0]

        if category in ade20k_classes + cityscape_classes + coco_stuff_classes + mapillary_classes:
            ious_in_4dataset[category] = ade20k_full_results[k]
        else:
            ious_not_in_4dataset[category] = ade20k_full_results[k]
    
    print('ious_not_in_4dataset: {} classes, mIoU: {}'.format( len(ious_not_in_4dataset), np.mean(list(ious_not_in_4dataset.values())) ))
    print('ious_in_4dataset: {} classes, mIoU: {}'.format( len(ious_in_4dataset), np.mean(list(ious_in_4dataset.values())) ))
    
    print('')
