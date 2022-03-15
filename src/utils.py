# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time, ast, torch, random



import numpy as np
import torch
from torch.autograd import Variable
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from graphviz import Digraph
from elasticsearch import Elasticsearch
import spacy
from math import (
    degrees, radians,
    sin, cos, asin, tan, atan, atan2, pi,
    sqrt, exp, log, fabs, log10, pow
)

from constants import (
    EARTH_MEAN_RADIUS,
    EARTH_MEAN_DIAMETER,
    EARTH_EQUATORIAL_RADIUS,
    EARTH_EQUATORIAL_METERS_PER_DEGREE,
    I_EARTH_EQUATORIAL_METERS_PER_DEGREE,
    HALF_PI,
    QUARTER_PI,
)


nlp = spacy.load("en_core_web_md")

def elastic_prediction(landmarks, sentences):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    for index, item in enumerate(landmarks):
        res = es.index(index='landmarks', id=index, body={'name': item})

    print("Testing...")
    # test3 = ['Redfish Point', 'pipeline area', 'obstructions','deep water area','platforms', 'southwest pass vermilion bay channel light 38', "light 36", "vermilion bay light 26","beacon 8" ]
    # trying NER from Spacy
    results = {}
    for sent in sentences:
        doc = nlp(sent)
        # tmp_pobjs = [e.text for e in doc if e.dep_ == 'obj']
        test10 = [e.text.lower() for e in doc.noun_chunks]
        tmp_ids = {}
        tmp_res = {}
        for te in test10:
            # print("--------------------------")
            # print(te)
            # print("Rsults:")

            # replace ITEM with the search query
            res = es.search(index='landmarks', body={'query': {'match': { 'name':{'query': te, 'fuzziness':'AUTO' }}}})
            for hit in res['hits']['hits']:
                tmp_ids[hit['_id']] = hit['_score']
                tmp_res[hit['_id']] = {'_score': hit['_score'],'_id': hit['_id'], 'name': hit['_source']['name']}
                # tmp_res[hit['_score']] =
                # input(hit)
                # print(hit['_source']['name'])
        tmp_ids = dict(reversed(sorted(tmp_ids.items(), key=lambda item: item[1])))
        results[sent] = {key : tmp_res[key] for key in list(tmp_ids.keys())[:3]}
    return results


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        #assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}


    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                #name = param_map[id(u)] if params is not None else ''
                #node_name = '%s\n %s' % (name, size_to_str(u.size()))
                node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')

            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDITEMS = ["img_id", "img_h", "img_w","num_boxes","t_num_boxes", "boxes",
              "features","names","t_boxes","t_names","box_order"]

# -122.44, 34.44 etc.
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
# can draw an item or list of items with names
def drawItem(name,image_path,points=None,pixels_bb=None):

    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 18)

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(e)
        return

    img_dim = [img.getbbox()[2],img.getbbox()[3]]

    if type(name) != list:
        name = [name]
        if pixels_bb is not None and type(pixels_bb) != list:
            pixels_bb = [pixels_bb]
        if points is not None and type(points) != list:
            points = [points]

    draw = ImageDraw.Draw(img)
    for id in range(len(name)):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        if points:
            draw.text(tuple((points[id]['points_x'][0],points[id]['points_y'][0])),name[id],fill=color,font=font)
            draw.line([tuple((x,y)) for x,y in zip(points[id]['points_x'],points[id]['points_y'])],fill=color)
        if pixels_bb:
            input(pixels_bb[id])
            draw.text(tuple((pixels_bb[id][0],pixels_bb[id][1])),name[id],fill=color,font=font)
            draw.rectangle(pixels_bb[id],outline=color)
        else:
            print("No pixels given. Give points x and y [points] or bounding box [pixels_bb]")
    img.show()


# returns dictionary with the name 'points_x' and 'points_y'
def generatePixel(scene_object,centre,zoom,img_dim, size=10):
    pix_pos = {}
    # depending on the geometry type generate pixels from the
    # raw GPS and save them. These are used for prediction and
    # for the generation of the attention masks
    if scene_object['g_type'] == 'Point':
        # print(img_dim)
        # print(zoom)
        # print(centre)
        # print([scene_object['coordinates'][1],scene_object['coordinates'][0]])
        tmppx = convertGeoToPixel(centre,[scene_object['coordinates'][1],scene_object['coordinates'][0]],zoom,img_dim)
        # input(tmppx)
        if tmppx[0] < 0 or tmppx[0] >= img_dim[0]  or tmppx[1] < 0 or tmppx[1] >= img_dim[1]:
            return None
        x_,y_ = midPointCircleDraw(tmppx[0],tmppx[1],size,img_dim)
        pix_pos['points_x'] = x_
        pix_pos['points_y'] = y_
        # tmp_land['points_x'] = tmppx[0]
        # tmp_land['points_y'] = tmppx[1]
        # tmp_land['all_att_pixels'] = [tmppx[0]-10,tmppx[1]-10,tmppx[0]+10,tmppx[1]+10]
    elif scene_object['g_type'] == 'LineString':
        tmppx = []
        for pnt in scene_object['coordinates']:
            tmp_px = convertGeoToPixel(centre,[pnt[1],pnt[0]],zoom,img_dim)

            if tmp_px[0] < 0 or tmp_px[0] >= img_dim[0]  or tmp_px[1] < 0 or tmp_px[1] >= img_dim[1]:
                continue
            tmppx.append(tmp_px)

        mask = Image.new('L', img_dim, color = 0)
        mask_draw=ImageDraw.Draw(mask)
        # highlighted_area = xy
        # if type == 'Point':
        new_tmppx = []
        if len(tmppx) >=2:
            mask_draw.line([(x_[0],x_[1]) for x_ in tmppx], fill = 255)
            width, height = mask.size
            pix2 = mask.load()
            for x in range(1,width):
                for y in range(1,height):
                    if pix2[x,y] == 255:
                        # input(pix2[x,y])
                        new_tmppx.append([float(x),float(y)])
            # mask.show()
        # input("??")
        pix_pos['points_x'] = [i[0] for i in new_tmppx]
        pix_pos['points_y'] = [i[1] for i in new_tmppx]

        # pix_pos['points_x'] = [i[0] for i in tmppx]
        # pix_pos['points_y'] = [i[1] for i in tmppx]
        # tmp_land['all_att_pixels'] = tmppx
        # draw.line(tmppx,fill=color)
        # dist_ = haversine(c[1],c[0],map['coordinates'][0][0],map['coordinates'][0][1])
    else:
        tmppx = []
        for side in scene_object['coordinates']:
            for pnt in side:
                tmp_px = convertGeoToPixel(centre,[pnt[1],pnt[0]],zoom,img_dim)
                # if tmp_px[0] < 0:
                #     tmp_px[0] = 0
                # if tmp_px[0] > img_dim[0]:
                #     tmp_px[0] = img_dim[0]
                # if tmp_px[1] < 0:
                #     tmp_px[1] = 0
                # if tmp_px[1] > img_dim[1]:
                #     tmp_px[1] = img_dim[1]
                if tmp_px[0] < 0 or tmp_px[0] >= img_dim[0]  or tmp_px[1] < 0 or tmp_px[1] >= img_dim[1]:
                    # print(tmp_px)
                    continue
                tmppx.append(tmp_px)
        mask = Image.new('L', img_dim, color = 0)
        mask_draw=ImageDraw.Draw(mask)
        # highlighted_area = xy
        # if type == 'Point':
        new_tmppx = []
        if len(tmppx) >=2:
            mask_draw.polygon([(x_[0],x_[1]) for x_ in tmppx], fill = 255)
            width, height = mask.size
            pix2 = mask.load()
            for x in range(1,width):
                for y in range(1,height):
                    if pix2[x,y] == 255:
                        # input(pix2[x,y])
                        new_tmppx.append([float(x),float(y)])
            # mask.show()
        # input("??")
        pix_pos['points_x'] = [i[0] for i in new_tmppx]
        pix_pos['points_y'] = [i[1] for i in new_tmppx]

    if len(pix_pos['points_x']) < 6 or len(pix_pos['points_y']) < 6:
        # print("less than 5 points")
        # print(pix_pos)
        # print(scene_object['name'])
        return None
    # tmp_regions[tmposm['category_id']] = tmp_land
    return pix_pos


# Implementing Mid-PoCircle Drawing
# Algorithm
def midPointCircleDraw(x_centre,y_centre, r,im_dim):
    x_points = []
    y_points = []
    x = r
    y = 0

    # Printing the initial poon the
    # axes after translation
    # print("(", x + x_centre, ", ",
    #            y + y_centre, ")",
    #            sep = "", end = "")
    if (x + x_centre) > 0 and (x + x_centre) < (im_dim[0]-1)  and (y + y_centre) > 0 and (y + y_centre) < (im_dim[1]-1):
        x_points.append(x + x_centre)
        y_points.append(y + y_centre)
    # When radius is zero only a single
    # powill be printed
    if (r > 0) :
        # print("(", x + x_centre, ", ",
        #           -y + y_centre, ")",
        #           sep = "", end = "")
        if (x + x_centre) > 0 and (x + x_centre) < (im_dim[0]-1)  and (-y + y_centre) > 0 and (-y + y_centre) < (im_dim[1]-1):
            x_points.append(x + x_centre)
            y_points.append(-y + y_centre)

        if (y + x_centre) > 0 and (y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 0 and (x + y_centre) < (im_dim[1]-1):
            x_points.append(y + x_centre)
            y_points.append(x + y_centre)

        if (-y + x_centre) > 0 and (-y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 0 and (x + y_centre) < (im_dim[1]-1):
            x_points.append(-y + x_centre)
            y_points.append(x + y_centre)
        #
        # print("(", -y + x_centre, ", ",
        #             x + y_centre, ")", sep = "")
    # Initialising the value of P
    P = 1 - r
    while (x > y) :

        y += 1

        # Mid-pois inside or on the
        # perimeter
        if (P <= 0):
            P = P + 2 * y + 1

        # Mid-pois outside the perimeter
        else:
            x -= 1
            P = P + 2 * y - 2 * x + 1

        # All the perimeter points have
        # already been printed
        if (x < y):
            break

        # Printing the generated poand its reflection
        # in the other octants after translation

        if (x + x_centre) > 1 and (x + x_centre) < (im_dim[0]-1)  and (y + y_centre) > 1 and (y + y_centre) < (im_dim[1]-1):
            x_points.append(x + x_centre)
            y_points.append(y + y_centre)
        # print("(", x + x_centre, ", ", y + y_centre,
        #                     ")", sep = "", end = "")

        if (-x + x_centre) > 1 and (-x + x_centre) < (im_dim[0]-1)  and (y + y_centre) > 1 and (y + y_centre) < (im_dim[1]- 1):
            x_points.append(-x + x_centre)
            y_points.append(y + y_centre)
        # print("(", -x + x_centre, ", ", y + y_centre,
        #                      ")", sep = "", end = "")

        if (x + x_centre) > 1 and (x + x_centre) < (im_dim[0]-1)  and (-y + y_centre) > 1 and (-y + y_centre) < (im_dim[1]- 1):
            x_points.append(x + x_centre)
            y_points.append(-y + y_centre)
        # print("(", x + x_centre, ", ", -y + y_centre,
        #                      ")", sep = "", end = "")
        # print("(", -x + x_centre, ", ", -y + y_centre,
        #                                 ")", sep = "")

        if (-x + x_centre) > 1 and (-x + x_centre) < (im_dim[0]-1)  and (-y + y_centre) > 1 and (-y + y_centre) < (im_dim[1]- 1):
            x_points.append(-x + x_centre)
            y_points.append(-y + y_centre)
        # If the generated pois on the line x = y then
        # the perimeter points have already been printed
        if (x != y) :

            if (y + x_centre) > 1 and (y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 1 and (x + y_centre) < (im_dim[1]- 1):
                x_points.append(y + x_centre)
                y_points.append(x + y_centre)
            # print("(", y + x_centre, ", ", x + y_centre,
            #                     ")", sep = "", end = "")

            if (-y + x_centre) > 1 and (-y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 1 and (x + y_centre) < (im_dim[1]- 1):
                x_points.append(-y + x_centre)
                y_points.append(x + y_centre)
            # print("(", -y + x_centre, ", ", x + y_centre,
            #                      ")", sep = "", end = "")

            if (y + x_centre) > 1 and (y + x_centre) < (im_dim[0]-1)  and (-x + y_centre) > 1 and (-x + y_centre) < (im_dim[1]- 1):
                x_points.append(y + x_centre)
                y_points.append(-x + y_centre)
            # print("(", y + x_centre, ", ", -x + y_centre,
            #                      ")", sep = "", end = "")

            if (-y + x_centre) > 1 and (-y + x_centre) < (im_dim[0]-1)  and (-x + y_centre) > 1 and (-x + y_centre) < (im_dim[1]- 1):
                x_points.append(-y + x_centre)
                y_points.append(-x + y_centre)
            # print("(", -y + x_centre, ", ", -x + y_centre,
            #                                 ")", sep = "")
    return x_points,y_points


def destination(point, distance, bearing):

    # print(point)
    lon1,lat1 = (radians(float(coord)) for coord in point)

    # print(distance," ",bearing)
    radians_bearing = radians(float(bearing))
    # print(radians_bearing)

    delta = float(distance) / EARTH_MEAN_RADIUS
    lat2 = asin(
        sin(lat1)*cos(delta) +
        cos(lat1)*sin(delta)*cos(radians_bearing)
    )
    numerator = sin(radians_bearing) * sin(delta) * cos(lat1)
    denominator = cos(delta) - sin(lat1) * sin(lat2)

    lon2 = lon1 + atan2(numerator, denominator)
    # print(type(lon2))
    lon2_deg = (degrees(lon2) + 540) % 360 - 180
    lat2_deg = degrees(lat2)

    return [lon2_deg,lat2_deg]

def getPointLatLng(x, y,centre_lon,centre_lat,_zoom,height,width):
    parallelMultiplier = cos(centre_lat * pi / 180)
    degreesPerPixelX = 360 / pow(2, _zoom + 8)
    degreesPerPixelY = 360 / pow(2, _zoom + 8) * parallelMultiplier
    pointLat = centre_lat - degreesPerPixelY * ( y - height / 2)
    pointLng = centre_lon + degreesPerPixelX * ( x  - width / 2)

    return (pointLat, pointLng)
# Calculating tile needed for converting GPS to pixels
# expects latlon like [37.0000, -122.2222]
def calculateTiles(latlon,zoom):

    lon_rad = radians(latlon[1]);
    lat_rad = radians(latlon[0]);
    n = pow(2.0, zoom);

    tileX = ((latlon[1] + 180) / 360) * n;
    tileY = (1 - (log(tan(lat_rad) + 1.0/cos(lat_rad)) / pi)) * n / 2.0;
    # print(" X: {}, Y: {}".format(tileX,tileY))
    return [tileX,tileY]

# Convert GPS to pixels. Need center of the map in GPS, lat/lon GPS, zoom level,
# dimension of the image.
def convertGeoToPixel(centre_, latlon, zoom, imgDim, adjust=False):
    # 	mapWidth = imgDim[0];
    # 	mapHeight = imgDim[1]
    #
    # double lon = lon_centre
    # double lat = lat_centre
    # double zoom = 6; # 6.5156731549786215 would be possible too
    if len(centre_) == 0:
        # print("New centre")
        centre_ = calculateTiles(latlon,zoom)
    point = calculateTiles(latlon,zoom)
    # print(centre_[0] - point[0])
    # print(imgDim)
    # print(zoom)
    if adjust:
        pix_x = imgDim[0]/2.13 - (centre_[0] - point[0])*256
        pix_y = imgDim[1]/2.5 - (centre_[1] - point[1])*256
    else:
        pix_x = imgDim[0]/2 - (centre_[0] - point[0])*256
        pix_y = imgDim[1]/2 - (centre_[1] - point[1])*256

    return [pix_x,pix_y]





def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def gen_chunks(reader, chunksize=100):
    """
    Chunk generator. Take a CSV `reader` and yield
    `chunksize` sized slices.
    """
    chunk = []
    for index, line in enumerate(tqdm(reader)):
        if (index % chunksize == 0 and index > 0):
            yield chunk
            del chunk[:]
        chunk.append(line)
    yield chunk

def load_det_obj_tsv(fname, topk=None):
    print(topk)
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, FIELDITEMS, delimiter="\t")
        if topk:
            chunk = topk
        else:
            chunk = 1000
        for it in gen_chunks(reader,  chunksize=chunk):
            # print(len(item[0]))
            # input(len(item))
            for i, item in enumerate(it):
                for key in ['img_h', 'img_w', 'num_boxes','t_num_boxes']:
                    item[key] = int(item[key])

                boxes = item['num_boxes']
                t_boxes = item['t_num_boxes']
                decode_config = [
                    ('boxes', (t_boxes, 4), np.float64),
                    ('t_boxes', (t_boxes, 4), np.float64),
                    ('features', (t_boxes, -1), np.float64),
                    ('names', (t_boxes, -1), np.dtype('<U100')),
                    ('t_names', (t_boxes, -1), np.dtype('<U100'))
                    # ('box_order', (t_boxes), np.float64)
                ]
                # input(decode_config)
                for key, shape, dtype in decode_config:
                    # print(key)
                    # print(item[key])
                    try:
                        item[key] = np.frombuffer(base64.b64decode(ast.literal_eval(item[key])), dtype=dtype)
                        item[key] = item[key].reshape(shape)
                        item[key].setflags(write=False)
                    except Exception as exc:
                        print(item[key])
                        print(item['names'])
                        print(exc)
                        input(key)

                data.append(item)
                if topk is not None and len(data) >= topk:
                    break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def giou_loss(output, target, bbox_inside_weights=None, bbox_outside_weights=None,
                transform_weights=None, batch_size=None):

    # if transform_weights is None:
    #     transform_weights = (1., 1., 1., 1.)

    if batch_size is None:
        batch_size = output.size(0)


    x1, y1, x2, y2 = output.t()[:,:]
    x1g, y1g, x2g, y2g = target.t()[:,:]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    # print(mask)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk)).mean(0) / output.size(0)
    giouk = ((1 - giouk)).mean(0) / output.size(0)

    return iouk, giouk


def iou_loss(output, target, reduction = 'mean'):
    # input(output)
    # input(output.shape)
    x1_t, y1_t, x2_t, y2_t = target.t()[:,:]
    x1_p, y1_p, x2_p, y2_p = output.t()[:,:]
    # print(x2_t)
    # print(x1_p)
    # print(torch.unique(x2_t < x1_p))
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t or x1_p > x2_p or y1_p > y2_p):
        # input(x2_t < x1_p)
        return None
    # make sure x2_p and y2_p are larger
    x2_p = torch.max(x1_p, x2_p)
    y2_p = torch.max(y1_p, y2_p)

    far_x = torch.min(x2_t, x2_p)
    near_x = torch.max(x1_t, x1_p)
    far_y = torch.min(y2_t, y2_p)
    near_y = torch.max(y1_t, y1_p)

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    # input(torch.mean(iou))
    if reduction != 'none':
            ret = torch.mean(iou) if reduction == 'mean' else torch.sum(iou)
    return 1 - ret
    # return loss

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """


    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box
    #
    # if (x1_p > x2_p) or (y1_p > y2_p):
    #     raise AssertionError(
    #         "Prediction box is malformed? pred box: {}".format(pred_box))
    # if (x1_t > x2_t) or (y1_t > y2_t):
    #     raise AssertionError(
    #         "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t or x1_p > x2_p or y1_p > y2_p):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou
