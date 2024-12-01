class ImageData:
    def __init__(self, filename, id, partition, **attributes):
        self.filename = filename
        self.id = int(id)
        self.partition = int(partition)
        self.image = None
        self.attributes = Attributes(**attributes)

class Attributes:
    def __init__(self, **kwargs):
        kwargs = {key: value == '1' for key, value in kwargs.items()}
        self.FIVE_O_CLOCK_SHADOW = kwargs.get('5_o_Clock_Shadow', False)
        self.ARCHED_EYEBROWS = kwargs.get('Arched_Eyebrows', False)
        self.ATTRACTIVE = kwargs.get('Attractive', False)
        self.BAGS_UNDER_EYES = kwargs.get('Bags_Under_Eyes', False)
        self.BALD = kwargs.get('Bald', False)
        self.BANGS = kwargs.get('Bangs', False)
        self.BIG_LIPS = kwargs.get('Big_Lips', False)
        self.BIG_NOSE = kwargs.get('Big_Nose', False)
        self.BLACK_HAIR = kwargs.get('Black_Hair', False)
        self.BLOND_HAIR = kwargs.get('Blond_Hair', False)
        self.BLURRY = kwargs.get('Blurry', False)
        self.BROWN_HAIR = kwargs.get('Brown_Hair', False)
        self.BUSHY_EYEBROWS = kwargs.get('Bushy_Eyebrows', False)
        self.CHUBBY = kwargs.get('Chubby', False)
        self.DOUBLE_CHIN = kwargs.get('Double_Chin', False)
        self.EYEGLASSES = kwargs.get('Eyeglasses', False)
        self.GOATEE = kwargs.get('Goatee', False)
        self.GRAY_HAIR = kwargs.get('Gray_Hair', False)
        self.HEAVY_MAKEUP = kwargs.get('Heavy_Makeup', False)
        self.HIGH_CHEEKBONES = kwargs.get('High_Cheekbones', False)
        self.MALE = kwargs.get('Male', False)
        self.MOUTH_SLIGHTLY_OPEN = kwargs.get('Mouth_Slightly_Open', False)
        self.MUSTACHE = kwargs.get('Mustache', False)
        self.NARROW_EYES = kwargs.get('Narrow_Eyes', False)
        self.NO_BEARD = kwargs.get('No_Beard', False)
        self.OVAL_FACE = kwargs.get('Oval_Face', False)
        self.PALE_SKIN = kwargs.get('Pale_Skin', False)
        self.POINTY_NOSE = kwargs.get('Pointy_Nose', False)
        self.RECEDING_HAIRLINE = kwargs.get('Receding_Hairline', False)
        self.ROSY_CHEEKS = kwargs.get('Rosy_Cheeks', False)
        self.SIDEBURNS = kwargs.get('Sideburns', False)
        self.SMILING = kwargs.get('Smiling', False)
        self.STRAIGHT_HAIR = kwargs.get('Straight_Hair', False)
        self.WAVY_HAIR = kwargs.get('Wavy_Hair', False)
        self.WEARING_EARRINGS = kwargs.get('Wearing_Earrings', False)
        self.WEARING_HAT = kwargs.get('Wearing_Hat', False)
        self.WEARING_LIPSTICK = kwargs.get('Wearing_Lipstick', False)
        self.WEARING_NECKLACE = kwargs.get('Wearing_Necklace', False)
        self.WEARING_NECKTIE = kwargs.get('Wearing_Necktie', False)
        self.YOUNG = kwargs.get('Young', False)
        
        
    def __call__(self):
        return [key for key, value in self.__dict__.items() if value]
