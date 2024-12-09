from enum import Enum

class ImageData:
    def __init__(self, filename, id, partition, **attributes):
        self.filename = filename
        self.id = int(id)
        self.partition = int(partition)
        self.image = None
        self.attributes = Attributes(**attributes)

    def __str__(self):
        return f"Image {self.filename} (ID: {self.id}, partition: {self.partition}) attributes: {self.attributes()}"


class Attribute(Enum):
    FIVE_O_CLOCK_SHADOW = '5_o_Clock_Shadow'
    ARCHED_EYEBROWS = 'Arched_Eyebrows'
    ATTRACTIVE = 'Attractive'
    BAGS_UNDER_EYES = 'Bags_Under_Eyes'
    BALD = 'Bald'
    BANGS = 'Bangs'
    BIG_LIPS = 'Big_Lips'
    BIG_NOSE = 'Big_Nose'
    BLACK_HAIR = 'Black_Hair'
    BLOND_HAIR = 'Blond_Hair'
    BLURRY = 'Blurry'
    BROWN_HAIR = 'Brown_Hair'
    BUSHY_EYEBROWS = 'Bushy_Eyebrows'
    CHUBBY = 'Chubby'
    DOUBLE_CHIN = 'Double_Chin'
    EYEGLASSES = 'Eyeglasses'
    GOATEE = 'Goatee'
    GRAY_HAIR = 'Gray_Hair'
    HEAVY_MAKEUP = 'Heavy_Makeup'
    HIGH_CHEEKBONES = 'High_Cheekbones'
    MALE = 'Male'
    MOUTH_SLIGHTLY_OPEN = 'Mouth_Slightly_Open'
    MUSTACHE = 'Mustache'
    NARROW_EYES = 'Narrow_Eyes'
    NO_BEARD = 'No_Beard'
    OVAL_FACE = 'Oval_Face'
    PALE_SKIN = 'Pale_Skin'
    POINTY_NOSE = 'Pointy_Nose'
    RECEDING_HAIRLINE = 'Receding_Hairline'
    ROSY_CHEEKS = 'Rosy_Cheeks'
    SIDEBURNS = 'Sideburns'
    SMILING = 'Smiling'
    STRAIGHT_HAIR = 'Straight_Hair'
    WAVY_HAIR = 'Wavy_Hair'
    WEARING_EARRINGS = 'Wearing_Earrings'
    WEARING_HAT = 'Wearing_Hat'
    WEARING_LIPSTICK = 'Wearing_Lipstick'
    WEARING_NECKLACE = 'Wearing_Necklace'
    WEARING_NECKTIE = 'Wearing_Necktie'
    YOUNG = 'Young'

class Attributes:
    def __init__(self, **kwargs):
        kwargs = {key: value == '1' for key, value in kwargs.items()}
        for attribute in Attribute:
            setattr(self, attribute.name, kwargs.get(attribute.value, False))
        
        
    def __call__(self):
        return [key for key, value in self.__dict__.items() if value]
