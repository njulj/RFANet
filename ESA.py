# This is the draft code of our ESA block, we provide it here for those who are eager to know
# the implementation details. The official version will be released later.


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
       if not padding and stride==1:
           padding = kernel_size // 2
       return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
       
class ESA(nn.Module):
     def __init__(self, n_feats, conv=default_conv):
         super(ESA, self).__init__()
         f = n_feats // 4
         self.conv1 = conv(n_feats, f, kernel_size=1)
         self.conv_f = conv(f, f, kernel_size=1)
         self.conv_max = conv(f, f, kernel_size=3, padding=1)
         self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
         self.conv3 = conv(f, f, kernel_size=3, padding=1)
         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
         self.conv4 = conv(f, n_feats, kernel_size=1)
         self.sigmoid = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)
  
     def forward(self, x, f):
         c1_ = (self.conv1(f))
         c1 = self.conv2(c1_)
         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
         v_range = self.relu(self.conv_max(v_max))
         c3 = self.relu(self.conv3(v_range))
         c3 = self.conv3_(c3)
         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', aligned_corners=False) 
         cf = self.conv_f(c1_)
         c4 = self.conv4(c3+cf)
         m = self.sigmoid(c4)
         
         return x * m
