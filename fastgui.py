import ipywidgets as widgets
from ipywidgets import FloatSlider, interact
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from IPython.display import display,clear_output, Javascript
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

style = {'description_width': 'initial'}

RED = '\033[31m'
BLUE = '\033[94m'
GREEN = '\033[92m'
BOLD   = '\033[1m'
ITALIC = '\033[3m'
RESET  = '\033[0m'

def dashboard_one():
    """GUI for first accordion window"""
    import torchvision
    try:
        import fastai2; fastver = fastai2.__version__
    except ImportError:
        fastver = 'fastai not found'
    try:
        import fastprogress; fastprog = fastprogress.__version__
    except ImportError:
        fastprog = 'fastprogress not found'
    try:
        import fastpages; fastp = fastpages.__version__
    except ImportError:
        fastp = 'fastpages not found'
    try:
        import nbdev; nbd = nbdev.__version__
    except ImportError:
        nbd = 'nbdev not found'

    print (BOLD +  RED + '>> fastGUI\n')
    button = widgets.Button(description='System', button_style='success')
    ex_button = widgets.Button(description='Explore', button_style='success')
    display(button)

    out = widgets.Output()
    display(out)

    def on_button_clicked_info(b):
        with out:
            clear_output()
            print(BOLD + BLUE + "fastai2 version: " + RESET + ITALIC + str(fastver))
            print(BOLD + BLUE + "nbdev version: " + RESET + ITALIC + str(nbd))
            print(BOLD + BLUE + "fastprogress version: " + RESET + ITALIC + str(fastprog))
            print(BOLD + BLUE + "fastpages version: " + RESET + ITALIC + str(fastp) + '\n')
            print(BOLD + BLUE + "python version: " + RESET + ITALIC + str(sys.version))
            print(BOLD + BLUE + "torchvision: " + RESET + ITALIC + str(torchvision.__version__))
            print(BOLD + BLUE + "torch version: " + RESET + ITALIC + str(torch.__version__))
            print(BOLD + BLUE + "\nCuda: " + RESET + ITALIC + str(torch.cuda.is_available()))
            print(BOLD + BLUE + "cuda version: " + RESET + ITALIC + str(torch.version.cuda))

    button.on_click(on_button_clicked_info)
def dashboard_two():
    """GUI for second accordion window"""
    dashboard_two.datas = widgets.ToggleButtons(
        options=['PETS', 'CIFAR', 'IMAGENETTE_160', 'IMAGEWOOF_160', 'MNIST_TINY'],
        description='Choose',
        value=None,
        disabled=False,
        button_style='info',
        tooltips=[''],
        style=style
    )
    display(dashboard_two.datas)

    button = widgets.Button(description='Explore', button_style='success')
    display(button)
    out = widgets.Output()
    display(out)
    def on_button_explore(b):
        with out:
            clear_output()
            ds_choice()
            show()
    button.on_click(on_button_explore)

#Helpers for dashboard two
def ds_choice():
    """Helper for dataset choices"""
    if dashboard_two.datas.value == 'PETS':
        ds_choice.source = untar_data(URLs.DOGS)
    elif dashboard_two.datas.value == 'CIFAR':
        ds_choice.source = untar_data(URLs.CIFAR)
    elif dashboard_two.datas.value == 'IMAGENETTE_160':
        ds_choice.source = untar_data(URLs.IMAGENETTE_160)
    elif dashboard_two.datas.value == 'IMAGEWOOF_160':
        ds_choice.source = untar_data(URLs.IMAGEWOOF_160)
    elif dashboard_two.datas.value == 'MNIST_TINY':
        ds_choice.source = untar_data(URLs.MNIST_TINY)

def plt_classes():
    ds_choice()
    print(BOLD + BLUE + "Dataset: " + RESET + BOLD + RED + str(dashboard_two.datas.value))
    """Helper for plotting classes in folder"""
    Path.BASE_PATH = ds_choice.source
    train_source = (ds_choice.source/'train/').ls().items
    print(BOLD + BLUE + "\n" + "No of classes: " + RESET + BOLD + RED + str(len(train_source)))

    num_l = []
    class_l = []
    for j, name in enumerate(train_source):
        fol = (ds_choice.source/name).ls().sorted()
        names = str(name)
        class_split = names.split('train')
        class_l.append(class_split[1])
        num_l.append(len(fol))

    y_pos = np.arange(len(train_source))
    performance = num_l

    fig = plt.figure(figsize=(7,7))
    plt.style.use('seaborn')
    plt.bar(y_pos, performance, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(y_pos, class_l, rotation=90)
    plt.ylabel('Images')
    plt.title('Images per Class')
    plt.show()

def display_images():
    """Helper for displaying images from folder"""
    train_source = (ds_choice.source/'train/').ls().items
    for i, name in enumerate(train_source):
        fol = (ds_choice.source/name).ls().sorted()
        fol_disp = fol[0:5]
        filename = fol_disp.items
        fol_tensor = [tensor(Image.open(o)) for o in fol_disp]
        img = fol_tensor[0]
        print(BOLD + BLUE + "Loc: " + RESET + str(name) + " " + BOLD + BLUE + "Number of Images: " + RESET +
              BOLD + RED + str(len(fol)))

        fig = plt.figure(figsize=(15,15))
        columns = 5
        rows = 1
        ax = []

        for i in range(columns*rows):
            for i, j in enumerate(fol_tensor):
                img = fol_tensor[i]    # create subplot and append to ax
                ax.append( fig.add_subplot(rows, columns, i+1))
                ax[-1].set_title("ax:"+str(filename[i]))  # set title
                plt.tick_params(bottom="on", left="on")
                plt.xticks([])
                plt.imshow(img)
        plt.show()
def browse_images():
    print(BOLD + BLUE + "Use slider to choose image" + RESET)
    ds_choice()
    items = get_image_files(ds_choice.source/'train/')
    n = len(items)
    def view_image(i):
        plt.imshow(Image.open(items[i]), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %s' % items[i])
        browse_images.img = items[i]
        plt.show()
    interact(view_image, i=(0,n-1))

def show():
    a = widgets.Output()
    b = widgets.Output()
    c = widgets.Output()
    with a:
        plt_classes()
    with b:
        display_images()
    with c:
        browse_images()
    view_one = VBox([a, c])
    view_two = HBox([view_one, b])
    display(view_two)

def aug_show():
    aug_button = widgets.Button(description='Augmentations', button_style='success')
    display(aug_button)
    aug_out = widgets.Output()
    display(aug_out)
    def on_aug_button(b):
        with aug_out:
            clear_output()
            j = widgets.Output()
            u = widgets.Output()
            with j:
                print(browse_images.img)
                display(Image.open(browse_images.img))
            with u:
                aug_dash()
            display(HBox([j, u]))
    aug_button.on_click(on_aug_button)

def aug_paras():
    """If augmentations is choosen show available parameters"""
    print(BOLD + BLUE + "Choose Augmentation Parameters: ")
    button_paras = widgets.Button(description='Confirm', button_style='success')

    aug_paras.hh = widgets.ToggleButton(value=False, description='Erase', button_style='info',
                                      style=style)
    aug_paras.cc = widgets.ToggleButton(value=False, description='Contrast', button_style='info',
                                      style=style)
    aug_paras.dd = widgets.ToggleButton(value=False, description='Rotate', button_style='info',
                                      style=style)
    aug_paras.ee = widgets.ToggleButton(value=False, description='Warp', button_style='info',
                                      style=style)
    aug_paras.ff = widgets.ToggleButton(value=False, description='Bright', button_style='info',
                                      style=style)
    aug_paras.gg = widgets.ToggleButton(value=False, description='DihedralFlip', button_style='info',
                                      style=style)
    aug_paras.ii = widgets.ToggleButton(value=False, description='Zoom', button_style='info',
                                      style=style)

    qq = widgets.HBox([aug_paras.hh, aug_paras.cc, aug_paras.dd, aug_paras.ee, aug_paras.ff, aug_paras.gg, aug_paras.ii])
    display(qq)
    display(button_paras)
    aug_par = widgets.Output()
    display(aug_par)
    def on_button_two_click(b):
        with aug_par:
            clear_output()
            aug_dash_choice()
    button_paras.on_click(on_button_two_click)

def aug():
    """Aug choice helper"""
    #Erase
    if aug_paras.hh.value == True:
            aug.b_max = FloatSlider(min=0,max=50,step=1,value=0, description='max count',
                                     orientation='horizontal', disabled=False)
            aug.b_pval = FloatSlider(min=0,max=1,step=0.1,value=0, description=r"$p$",
                                     orientation='horizontal', disabled=False)
            aug.b_asp = FloatSlider(min=0.1,max=5, step=0.1, value=0.3, description=r'$aspect$',
                                     orientation='horizontal', disabled=False)
            aug.b_len = FloatSlider(min=0.1,max=5, step=0.1, value=0.3, description=r'$sl$',
                                     orientation='horizontal', disabled=False)
            aug.b_ht = FloatSlider(min=0.1,max=5, step=0.1, value=0.3, description=r'$sh$',
                                     orientation='horizontal', disabled=False)
            aug.erase_code = 'this is ERASE on'
    if aug_paras.hh.value == False:
            aug.b_max = FloatSlider(min=0,max=10,step=1,value=0, description='max count',
                                     orientation='horizontal', disabled=True)
            aug.b_pval = FloatSlider(min=0,max=1,step=0.1,value=0, description='p',
                                     orientation='horizontal', disabled=True)
            aug.b_asp = FloatSlider(min=0.1,max=1.7,value=0.3, description='aspect',
                                     orientation='horizontal', disabled=True)
            aug.b_len = FloatSlider(min=0.1,max=1.7,value=0.3, description='length',
                                     orientation='horizontal', disabled=True)
            aug.b_ht = FloatSlider(min=0.1,max=1.7,value=0.3, description='height',
                                     orientation='horizontal', disabled=True)
            aug.erase_code = 'this is ERASE OFF'
    #Contrast
    if aug_paras.cc.value == True:
            aug.b1_max = FloatSlider(min=0,max=0.9,step=0.1,value=0.2, description='max light',
                                  orientation='horizontal', disabled=False)
            aug.b1_pval = FloatSlider(min=0,max=1.0,step=0.05,value=0.75, description='p',
                                  orientation='horizontal', disabled=False)
            aug.b1_draw = FloatSlider(min=0,max=100,step=1,value=1, description='draw',
                                  orientation='horizontal', disabled=False)
    else:
            aug.b1_max = FloatSlider(min=0,max=0.9,step=0.1,value=0, description='max light',
                                  orientation='horizontal', disabled=True)
            aug.b1_pval = FloatSlider(min=0,max=1.0,step=0.05,value=0.75, description='p',
                                  orientation='horizontal', disabled=True)
            aug.b1_draw = FloatSlider(min=0,max=100,step=1,value=1, description='draw',
                                  orientation='horizontal', disabled=True)
    #Rotate
    if aug_paras.dd.value == True:
            aug.b2_max = FloatSlider(min=0,max=10,step=1,value=0, description='max degree',
                                  orientation='horizontal', disabled=False)
            aug.b2_pval = FloatSlider(min=0,max=1,step=0.1,value=0.5, description='p',
                                  orientation='horizontal', disabled=False)
    else:
            aug.b2_max = FloatSlider(min=0,max=10,step=1,value=0, description='max degree',
                                  orientation='horizontal', disabled=True)
            aug.b2_pval = FloatSlider(min=0,max=1,step=0.1,value=0, description='p',
                                  orientation='horizontal', disabled=True)
    #Warp
    if aug_paras.ee.value == True:
            aug.b3_mag = FloatSlider(min=0,max=10,step=1,value=0, description='magnitude',
                                  orientation='horizontal', disabled=False)
            aug.b3_pval = FloatSlider(min=0,max=1,step=0.1,value=0, description='p',
                                  orientation='horizontal', disabled=False)
    else:
            aug.b3_mag = FloatSlider(min=0,max=10,step=1,value=0, description='magnitude',
                                  orientation='horizontal', disabled=True)
            aug.b3_pval = FloatSlider(min=0,max=10,step=1,value=0, description='p',
                                  orientation='horizontal', disabled=True)
    #Bright
    if aug_paras.ff.value == True:
            aug.b4_max = FloatSlider(min=0,max=10,step=1,value=0, description='max light',
                                  orientation='horizontal', disabled=False)
            aug.b4_pval = FloatSlider(min=0,max=1,step=0.1,value=0, description='p',
                                  orientation='horizontal', disabled=False)
    else:
            aug.b4_max = FloatSlider(min=0,max=10,step=1,value=0, description='max_light',
                                  orientation='horizontal', disabled=True)
            aug.b4_pval = FloatSlider(min=0,max=1,step=0.1,value=0, description='p',
                                  orientation='horizontal', disabled=True)
    #DihedralFlip
    if aug_paras.gg.value == True:
            aug.b5_pval = FloatSlider(min=0,max=1,step=0.1, description='p',
                                     orientation='horizontal', disabled=False)
            aug.b5_draw = FloatSlider(min=0,max=7,step=1, description='p',
                                     orientation='horizontal', disabled=False)
    else:
            aug.b5_pval = FloatSlider(min=0,max=1,step=0.1, description='p',
                                     orientation='horizontal', disabled=True)
            aug.b5_draw = FloatSlider(min=0,max=7,step=1, description='p',
                                     orientation='horizontal', disabled=True)
    #Zoom
    if aug_paras.ii.value == True:
            aug.b6_zoom = FloatSlider(min=1,max=5,step=0.1, description='max_zoom',
                                     orientation='horizontal', disabled=False)
            aug.b6_pval = FloatSlider(min=0,max=1,step=0.1, description='p',
                                     orientation='horizontal', disabled=False)
    else:
            aug.b6_zoom = FloatSlider(min=1,max=5,step=0.1, description='max_zoom',
                                     orientation='horizontal', disabled=True)
            aug.b6_pval = FloatSlider(min=0,max=1,step=1, description='p',
                                     orientation='horizontal', disabled=True)

def aug_dash_choice():
    """Augmention parameter display helper"""
    button_aug_dash = widgets.Button(description='View', button_style='success')
    item_erase_val= widgets.HBox([aug.b_max, aug.b_pval, aug.b_asp, aug.b_len, aug.b_ht])
    item_erase = widgets.VBox([aug_paras.hh, item_erase_val])

    item_contrast_val = widgets.HBox([aug.b1_max, aug.b1_pval, aug.b1_draw])
    item_contrast = widgets.VBox([aug_paras.cc, item_contrast_val])

    item_rotate_val = widgets.HBox([aug.b2_max, aug.b2_pval])
    item_rotate = widgets.VBox([aug_paras.dd, item_rotate_val])

    item_warp_val = widgets.HBox([aug.b3_mag, aug.b3_pval])
    item_warp = widgets.VBox([aug_paras.ee, item_warp_val])

    item_bright_val = widgets.HBox([aug.b4_max, aug.b4_pval])
    item_bright = widgets.VBox([aug_paras.ff, item_bright_val])

    item_dihedral_val = widgets.HBox([aug.b5_pval, aug.b5_draw])
    item_dihedral = widgets.VBox([aug_paras.gg, item_dihedral_val])

    item_zoom_val = widgets.HBox([aug.b6_zoom, aug.b6_pval])
    item_zoom = widgets.VBox([aug_paras.ii, item_zoom_val])

    items = [item_erase, item_contrast, item_rotate, item_warp, item_bright, item_dihedral, item_zoom]
    dia = Box(items, layout=Layout(
                    display='flex',
                    flex_flow='column',
                    flex_grow=0,
                    flex_wrap='wrap',
                    border='solid 1px',
                    align_items='flex-start',
                    align_content='flex-start',
                    justify_content='space-between',
                    width='flex'
                    ))
    display(dia)
    display(button_aug_dash)
    aug_dash_out = widgets.Output()
    display(aug_dash_out)
    def on_button_two(b):
        with aug_dash_out:
            clear_output()
            print(browse_images.img)
    button_aug_dash.on_click(on_button_two)

def aug_dash():
    """GUI for augmentation dashboard"""
    tg = widgets.Button(description='Pad', disabled=True, button_style='info')
    aug_dash.pad = widgets.ToggleButtons(value='Reflection', options=['Zeros', 'Reflection', 'Border'], description='',
                                         button_style='info',style=style, layout=Layout(width='auto'))
    th = widgets.Button(description='ResizeMethod', disabled=True, button_style='warning')
    aug_dash.rzm = widgets.ToggleButtons(value='Squish', options=['Squish', 'Pad', 'Crop'], description='',
                                         button_style='warning', style=style, layout=Layout(width='auto'))
    ti = widgets.Button(description='Resize', disabled=True, button_style='primary')
    aug_dash.res = widgets.ToggleButtons(value='128', options=['28', '64', '128', '194', '254'], description='',
                                         button_style='primary', style=style, layout=Layout(width='auto'))
    aug_paras.hh = widgets.ToggleButton(value=False, description='Erase', button_style='info',
                                      style=style)
    aug_paras.cc = widgets.ToggleButton(value=False, description='Contrast', button_style='info',
                                      style=style)
    aug_paras.dd = widgets.ToggleButton(value=False, description='Rotate', button_style='info',
                                      style=style)
    aug_paras.ee = widgets.ToggleButton(value=False, description='Warp', button_style='info',
                                      style=style)
    aug_paras.ff = widgets.ToggleButton(value=False, description='Bright', button_style='info',
                                      style=style)
    aug_paras.gg = widgets.ToggleButton(value=False, description='DihedralFlip', button_style='info',
                                      style=style)
    aug_paras.ii = widgets.ToggleButton(value=False, description='Zoom', button_style='info',
                                      style=style)

    qq = widgets.HBox([aug_paras.hh, aug_paras.cc, aug_paras.dd, aug_paras.ee, aug_paras.ff, aug_paras.gg, aug_paras.ii])

    it2 = [tg, aug_dash.pad]
    it3 = [th, aug_dash.rzm]
    it4 = [ti, aug_dash.res]
    il = widgets.HBox(it2)
    ik = widgets.HBox(it3)
    ij = widgets.HBox(it4)
    ir = widgets.VBox([il, ik, ij])
    display(ir)
    print(BOLD + BLUE + "Choose Augmentation Parameters: ")
    display(qq)
    aug_img()

def aug_img():
    aug_img_b = widgets.Button(description='Confirm', button_style='success')
    display(aug_img_b)
    aug_img_out = widgets.Output()
    display(aug_img_out)
    def aug_img_(b):
        with aug_img_out:
            clear_output()
            aug_img = browse_images.img
            imgt = Image.open(aug_img)
            h1, w1 = imgt.shape
            pil_img = PILImage(PILImage.create(aug_img).resize((w1,h1))) #flip
            print(BOLD + BLUE + 'Size:' + RED + aug_dash.res.value + BLUE + ' ResizeMode:' + RED +
                  aug_dash.rzm.value + BLUE + ' Padding:' + RED + aug_dash.pad.value + RESET)
            if aug_dash.rzm.value == 'Pad': method = ResizeMethod.Pad
            if aug_dash.rzm.value == 'Squish': method = ResizeMethod.Squish
            if aug_dash.rzm.value == 'Crop': method = ResizeMethod.Crop
            if aug_dash.pad.value == 'Zeros': pad = PadMode.Zeros
            if aug_dash.pad.value == 'Border': pad = PadMode.Border
            if aug_dash.pad.value == 'Reflection': pad = PadMode.Reflection
            rsz = Resize(int(aug_dash.res.value), method=method, pad_mode=pad)
            display(show_image(rsz(pil_img)))
    aug_img_b.on_click(aug_img_)

def display_ui():
    """ Display tabs for visual display"""
    out1a = widgets.Output()
    out1 = widgets.Output()
    out2 = widgets.Output()
    data1a = pd.DataFrame(np.random.normal(size = 50))
    data1 = pd.DataFrame(np.random.normal(size = 100))
    data2 = pd.DataFrame(np.random.normal(size = 150))

    with out1a: #info
        clear_output()
        dashboard_one()

    with out1: #data
        clear_output()
        dashboard_two()

    with out2: #augmentation
        clear_output()
        aug_show()

    display_ui.tab = widgets.Tab(children = [out1a, out1, out2])
    display_ui.tab.set_title(0, 'Info')
    display_ui.tab.set_title(1, 'Data')
    display_ui.tab.set_title(2, 'Augmentation')
    display(display_ui.tab)
