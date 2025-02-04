def IntersectionOverUnion(y_label,y_pred):
    
    # overlap (Logical AND)
    overlap = y_label*y_pred
    union = y_label+y_pred

    IOU = overlap.sum()/float(union.sum()+ 1e-9)
    return IOU