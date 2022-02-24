from multiprocessing import Process, Queue
import pypangolin as pangolin
import numpy as np
import OpenGL.GL as gl


class Display:

    w = 1024
    h = 768
    f = 420

    def __init__(self, w, h):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True # a daemon is a computer program that runs in the background
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init()
        while True:
            self.viewer_refresh(q)
    
    def viewer_init(self):
        pangolin.CreateWindowAndBind("Map", self.w, self.h) #create window
        gl.glEnable(gl.GL_DEPTH_TEST) # enable depth

        self.s_cam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.w, self.h, self.f, self.f, self.w//2, self.h//2, 0.2, 1000),
            pangolin.ModelViewLookAt(-2, -2, -2, 
                                      0,  0,  0, 
                                      0, -1, 0)
        )
        self.handler = pangolin.Handler3D(self.s_cam)

        # Create window (for interaction)
        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(pangolin.Attach(0.), pangolin.Attach(1.), pangolin.Attach.Pix(320), pangolin.Attach(1.), -self.w/self.h)
        self.d_cam.SetHandler(self.handler)
        self.d_cam.Activate(self.s_cam)

# def main():
#     win = pango.CreateWindowAndBind("pySimpleDisplay", 640, 480)
#     glEnable(GL_DEPTH_TEST)

#     pm = pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
#     mv = pango.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pango.AxisY)
#     s_cam = pango.OpenGlRenderState(pm, mv)

#     ui_width = 180

#     handler = pango.Handler3D(s_cam)
#     d_cam = (
#         pango.CreateDisplay()
#         .SetBounds(
#             pango.Attach(0),
#             pango.Attach(1),
#             pango.Attach.Pix(ui_width),
#             pango.Attach(1),
#             -640.0 / 480.0,
#         )
#         .SetHandler(handler)
#     )

#     pango.CreatePanel("ui").SetBounds(
#         pango.Attach(0), pango.Attach(1), pango.Attach(0), pango.Attach.Pix(ui_width)
#     )
#     var_ui = pango.Var("ui")
#     var_ui.a_Button = False
#     var_ui.a_double = (0.0, pango.VarMeta(0, 5))
#     var_ui.an_int = (2, pango.VarMeta(0, 5))
#     var_ui.a_double_log = (3.0, pango.VarMeta(1, 1e4, logscale=True))
#     var_ui.a_checkbox = (False, pango.VarMeta(toggle=True))
#     var_ui.an_int_no_input = 2
#     var_ui.a_str = "sss"

#     ctrl = -96
#     pango.RegisterKeyPressCallback(ctrl + ord("a"), a_callback)

#     while not pango.ShouldQuit():
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#         if var_ui.a_checkbox:
#             var_ui.an_int = var_ui.a_double

#         var_ui.an_int_no_input = var_ui.an_int

#         d_cam.Activate(s_cam)
#         pango.glDrawColouredCube()
#         pango.FinishFrame()


# if __name__ == "__main__":
#     main()