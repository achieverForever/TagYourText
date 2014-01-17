#!/usr/bin/env python
# -*- coding: utf-8 -*-  
import wx
import time
import wx.stc as stc
import os
import os.path
import matplotlib.pyplot as plt
import jieba.posseg as pseg
import jieba
import jieba.analyse 
import matplotlib
import sys
from pylab import *


import numpy as np
import utils
from codecs import open
from subprocess import Popen, PIPE
import subprocess

from utils import VAR_DICT_FILE 
from utils import VAR_NEW_BOW_FILE 
from utils import VAR_TOPNWORDS_FILE 
from utils import VAR_SYS_ENCODING 
from utils import VAR_STOPWORD_FILE 
from utils import VAR_THETA_FILE_RESULT
from utils import VAR_MODEL_FILE  
from utils import VAR_NUM_WORD_TOPIC_FILE
from utils import VAR_NUM_DOC_TOPIC_FILE
from utils import VAR_TOTAL_WORDS_PER_TOPIC_FILE
from utils import VAR_NEW_PATH_FILE
from preprocess import Preprocessor


class MyPanel(wx.Panel):
    def __init__(self, parent, image_file='Pie.png'):
        wx.Panel.__init__(self, parent, id=-1)
        try:
            imagefile=image_file
            image = wx.Image(imagefile)
            temp = image.ConvertToBitmap()
            size = temp.GetWidth(),temp.GetHeight()
            wx.StaticBitmap(self, -1, temp)            
        except IOError:
            print 'Image file %s not found' % image_file
            raise SystemExi

class MyFrame(wx.Frame):

    title = "Show more detial"

    phasesList = [u"文学", u"体育", u"金融","IT"]
    def __init__(self,filenames="",label="",count=0):
        wx.Frame.__init__(self, wx.GetApp().TopWindow, title=self.title,size=(720,400))
        self.filename=filenames
        self.panel1=wx.Panel(self)
        self.label1 = wx.StaticText(self.panel1, -1, label=u"所属类别为：  "+label)
        # self.label2=wx.StaticText(self.panel1,-1,label=u"结果不对我要更正:")
        # self.label1.SetForegroundColour("blue")
        # self.label2.SetForegroundColour("blue")
        self.content2 = wx.TextCtrl(self.panel1, style=wx.TE_MULTILINE)


        with open(filenames, encoding='utf-8') as f:
            doc =f.read()
        self.content2.SetValue(doc)

        self.panel2=MyPanel(self.panel1,"Pie.png")
        # self.content3=wx.ComboBox(self.panel1, choices=self.phasesList)
        # self.content3.Bind(wx.EVT_COMBOBOX, self.onCombo)
        # self.button2=wx.Button(self.panel1,label=u"更改")

        #
        self.hbox1=wx.BoxSizer()
        self.hbox1.Add(self.label1,proportion=0,flag=wx.EXPAND|wx.LEFT,border=10)
        self.hbox2=wx.BoxSizer()
        self.hbox2.Add(self.content2,proportion=1,flag=wx.EXPAND|wx.LEFT,border=10)
        self.hbox2.Add(self.panel2,proportion=0,flag=wx.EXPAND|wx.RIGHT,border=10)
        # self.hbox3=wx.BoxSizer()
        # self.hbox3.Add(self.label2,proportion=0,flag=wx.LEFT,border=10)
        # self.hbox3.Add(self.content3,proportion=0,flag=wx.LEFT|wx.RIGHT,border=5)
        # self.hbox3.Add(self.button2,proportion=0,flag=wx.RIGHT,border=10)

        self.vbox1=wx.BoxSizer(wx.VERTICAL)
        self.vbox1.Add(self.hbox1,proportion=0,flag=wx.EXPAND|wx.TOP|wx.BOTTOM,border=5 )
        self.vbox1.Add(self.hbox2,proportion=1,flag=wx.EXPAND|wx.TOP|wx.BOTTOM,border=5 )
        # self.vbox1.Add(self.hbox3,proportion=0,flag=wx.EXPAND|wx.TOP|wx.BOTTOM,border=5 )
        self.panel1.SetSizer(self.vbox1) 
    def onCombo(self, event):
        """
        """
        phaseSelection = self.content3.GetValue()
        # print phaseSelection

class Frame(wx.Frame):   

    def __init__(self): #3
        """ 

        the Constructor wx.Frame(parent, id=-1, title="", pos=wx.DefaultPosition,
            size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE,
            name="frame")

        """
        title="Tag Your text"
        wx.Frame.__init__(self, None, title=title,
                          size=(800,600))        
        self.panel =wx.Panel(self)
        self.panel.SetBackgroundColour((0,205,213))
        self.CreateListctr()
        self.CreateTextBox()
        self.CreateButton()
        self.createLayout()
        self.BindEvent()
        self.InitStatusBar()
        self.CreateMenuBar()
        self.statusBasPrint(u"加载成功")
        self.dirname = None

#init the Frame
    def CreateListctr(self):
        
        self.list = wx.ListCtrl(self.panel, -1, style=wx.LC_REPORT)
        self.list.InsertColumn(0, u"文件名", width=100)
        self.list.InsertColumn(1, u"主题分布", width=400)
        self.list.InsertColumn(2,u"类别",width=100)
        self.list.InsertColumn(3, u"分布注释", wx.LIST_FORMAT_RIGHT, 80)


    def CreateTextBox(self):
        '''
            create TextBox
            the contents2 is the ColorTextBox class which is the Subclass of the wx.stc.StyledTextCtrl
            we use its write function to tag the highlight words
        '''
        self.contents1=wx.TextCtrl(self.panel)  #add the texttctl componment
        # self.contents2=ColorTextBox(self.panel)
        # self.contents2.SetWrapMode(stc.STC_WRAP_WORD)

    def BindEvent(self):
        '''
            bind the button
        '''
        self.button1.Bind(wx.EVT_BUTTON, self.category)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED,self.Onclick)

    def CreateButton(self):
        '''
            create the button
        '''
        self.button1=wx.Button(self.panel,label=u'分类')

    def createLayout(self):
        '''
            BoxSizer is the Layout component 
            vbox is the Vertical
            hbox is the Horizontal
        '''
        self.hbox=wx.BoxSizer()
        self.hbox.Add(self.contents1,proportion=1,flag=wx.EXPAND|wx.LEFT,border=30)
        self.hbox.Add(self.button1,proportion=0,flag=wx.RIGHT,border=30)


        self.vbox=wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.hbox,proportion=0,flag=wx.EXPAND|wx.TOP|wx.BOTTOM,border=30 )
        self.vbox.Add(self.list,proportion=1,flag=wx.EXPAND|wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT,border=30)
        self.panel.SetSizer(self.vbox) 

    def builtdata(self,datalist):
        i=0
        newlist=[]
        for data1 in datalist:
            su=0
            a=data1[:3]
            # print a
            for k in a:
                su+=k[1]
            a1=[(j[0],j[1]/su*100) for j in a]
            sts=''
            for d in a1:
                fds = '{0:.1f}%\t'.format(d[1])
                sts+=d[0] + ": " + fds + "     "
            newlist.append([str(i),sts,a1[0][0],u"双击查看详情"])
            i+=1
        return newlist

    def category(self,event):

        if self.dirname is None:
            return

        start = time.time()
        self.statusBasPrint(u"系统正在疯狂运算中，请稍候...")

        Preprocessor.preprocess(self.dirname)

        self.id2path = utils.parse_path_file(VAR_NEW_PATH_FILE)

        self.statusBasPrint('C++ classifier is running...')

        # p = subprocess.call('Utils.exe', stdout=PIPE, stdin=PIPE, stderr=PIPE)
        p = subprocess.call('Utils.exe')

        self.statusBasPrint('C++ classifier has finished')

        theta = np.loadtxt(VAR_THETA_FILE_RESULT)

        # return as a list [ [(name, probability), (name, probability), ...], ...]
        self.result = utils.get_topic_dist(theta)
        self.packages=self.builtdata(self.result)
        for i in self.packages:
            index = self.list.InsertStringItem(sys.maxint, i[0])
            self.list.SetStringItem(index, 1, i[1])
            self.list.SetStringItem(index, 2, i[2])
            self.list.SetStringItem(index,3,i[3])
            #max(dict.iterkeys(),key=lambda k:dict[k])
        self.statusBasPrint(u"文档分类完成！")
        elapsed = (time.time() - start)
        self.Printtime(str(elapsed)+'s')





    def Onclick(self, event):
        data=self.result[int(event.GetText())][:3]
        su=0
        for da in data:
            su+=da[1]
        fracs=[i[1]/su*100 for i in data]
        # print fracs
        mylabel=[i[0] for i in data]
        # print mylabel 

        self.id = int(event.GetText())

        if os.path.isfile('Pie.png'):
            os.remove('Pie.png')

        self.createPie(mylabel,fracs)
        MyFrame(self.id2path[self.id],mylabel[0]).Show()
    def createPie(self,mylabels=['it', u'你妹', 'EQ'],fracs=[40, 30, 30]):

        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  #设置缺省字体
        matplotlib.rcParams['toolbar'] = 'None'       

        plt.ioff()

        fig = figure(3, figsize=(3,3))
        mycolors=['red', 'yellow', 'green']
        pie(fracs,labels=mylabels,colors=mycolors)
        savefig('Pie.png',transparent=False, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
#Ok
    def InitStatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(2)
        self.statusbar.SetStatusWidths([-4,-1])  


    def statusBasPrint(self,strs):
        self.statusbar.SetStatusText(u''+strs, 0)

    def Printtime(self,strs):
        self.statusbar.SetStatusText(u'用时  '+strs,1)
    def MenuData(self):
        '''
                   menu data
        '''

        return [("&Menu", (                       
                           ("Open", u"打开", self.OnOpen),
                           ("&About", u"关于本软件", self.Onabout),                                  #
                           ("&Quit", u"退出", self.OnCloseWindow)))
               ] 
    def CreateMenuBar(self):
        '''
        createMunBar
        '''
        menuBar = wx.MenuBar()
        for eachMenuData in self.MenuData():
            menuLabel = eachMenuData[0]
            menuItems = eachMenuData[1]
            menuBar.Append(self.CreateMenu(menuItems), menuLabel) 
        self.SetMenuBar(menuBar)
    def CreateMenu(self, menuData):
        '''
        create the munu
        '''
        menu = wx.Menu()
        for eachItem in menuData:
            if len(eachItem) == 2:
                label = eachItem[0]
                subMenu = self.CreateMenu(eachItem[1])
                menu.AppendMenu(wx.NewId(), label, subMenu) #
            else:
                self.CreateMenuItem(menu, *eachItem)
        return menu
    def CreateMenuItem(self, menu, label, status, handler, kind = wx.ITEM_NORMAL):
        '''topic
        create the item in the menu
        '''
        if not label:
            menu.AppendSeparator()
            return
        menuItem = menu.Append(-1, label, status, kind)
        self.Bind(wx.EVT_MENU, handler,menuItem)


#event handler of the menu
    def OnOpen(self, event):
        dlg = wx.DirDialog(self, "Please choose your directory which contains txts:", \
                          style=1)
        if dlg.ShowModal() == wx.ID_OK:
            self.dirname = dlg.GetPath()
            self.contents1.SetValue(self.dirname)

    def Onabout(self,event):
        mess='Tag Your Text Version 1.0 \nAuthor: \
        GuoWeiCheng, WangJunSheng, HuangHongJie, LinJingPei, GuoBaoRong and GuoMengLu!'
        dlg = wx.MessageDialog(None,mess,
                      'What about !', wx.OK | wx.ICON_INFORMATION)
        result = dlg.ShowModal()
        dlg.Destroy()
    def OnCloseWindow(self, event):
        self.Destroy()

     
class App(wx.App):

    def OnInit(self):
        self.frame=Frame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

def main():  # main

    app = App(redirect=False)
    app.MainLoop()

if __name__ == '__main__':
     main()

