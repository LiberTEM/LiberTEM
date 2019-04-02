import numpy as np


class EventHandler(DM.Py_ScriptObject):

    def HandleComponentChangedEvent(self, img_disp_event_flags, img_disp, 
            component_change_flag, component_disp_change_flags, component):
        print("changed")


z = np.zeros((100, 100))

image = DM.CreateImage(z)

doc = DM.NewImageDocument("test document")
d = doc.AddImageDisplay(image, 1)
c = d.AddNewComponent(5, 40, 40, 60, 60)
c.SetForegroundColor(1, 0, 0)

listener = EventHandler()

listener.ImageDisplayHandleComponentChangedEvent(d, 'pythonplugin')

doc.Show()
