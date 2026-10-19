class Controller:
    # NES standard pad: A, B, Select, Start, Up, Down, Left, Right
    def __init__(self):
        self.state = 0
        self.shift = 0
        self.strobe = 0

    def set_buttons(self, names):
        bits = {'A':0, 'B':1, 'SELECT':2, 'START':3,
                'UP':4, 'DOWN':5, 'LEFT':6, 'RIGHT':7}
        self.state = 0
        for name in names:
            b = bits.get(name.upper())
            if b is not None:
                self.state |= 1 << b
        if self.strobe:
            self.shift = self.state

    def write(self, value):
        old = self.strobe
        self.strobe = value & 1
        if self.strobe or old != self.strobe:
            self.shift = self.state

    def read(self):
        if self.strobe:
            return (self.state & 1) | 0x40
        value = (self.shift & 1) | 0x40
        self.shift = (self.shift >> 1) | 0x80
        return value
