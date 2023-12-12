from src.utils import get_iou_value


def match_person_with_moto(person_bboxes, moto_bboxes, min_iou_value=0.0):
    person_to_moto = {}
    
    for i, person_bbox in enumerate(person_bboxes):
        max_iou, max_idx = 0.0, -1
        
        for j, moto_bbox in enumerate(moto_bboxes):
            curr_iou = get_iou_value(person_bbox, moto_bbox)
            if curr_iou > max_iou:
                max_iou = curr_iou
                max_idx = j
        
        if max_iou > min_iou_value:
            person_to_moto[i] = (moto_bboxes[max_idx], max_iou, max_idx)
            
    return person_to_moto


def match_moto_with_person(moto_bboxes, person_to_moto):
    moto_to_person = {}
    
    for i, _ in enumerate(moto_bboxes):
        moto_to_person[i] = []
        for j, (_, _, matched_moto_id) in person_to_moto.items():
            if i == matched_moto_id:
                moto_to_person[i].append(j)
        
    return moto_to_person


def match_motorcycles_and_pilots(person_boxes, moto_boxes):
    matched_person_to_moto = match_person_with_moto(person_boxes, moto_boxes)
    matched_moto_to_person = match_moto_with_person(moto_boxes, matched_person_to_moto)

    return matched_moto_to_person
