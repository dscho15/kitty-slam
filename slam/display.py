from multiprocessing import Process, Queue
import pypangolin as pangolin
import numpy as np
import OpenGL.GL as gl
import time
import logging

class Display:

    w = 1024
    h = 768
    f = 2000

    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True # a daemon is a computer program that runs in the background
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init()
        while not pangolin.ShouldQuit():
            if not q.empty():
                self.viewer_refresh(q)
    
    def viewer_init(self):
        # Create window
        pangolin.CreateWindowAndBind("Map", self.w, self.h)
        # Enable depth
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Source Camera
        self.s_cam = pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(self.w, self.h, self.f, self.f, self.w//2, self.h//2, 0.2, 1000), 
                                                pangolin.ModelViewLookAt(0, -30, -30, 0,  0,  0, 0, -1, 0))
        # 3D Handler
        self.handler = pangolin.Handler3D(self.s_cam)
        # Create window (for interaction)
        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(pangolin.Attach(0.), pangolin.Attach(1.), pangolin.Attach.Pix(320), pangolin.Attach(1.), -self.w/self.h)
        self.d_cam.SetHandler(self.handler)
        self.d_cam.Activate(self.s_cam)
    
    def viewer_refresh(self, q: Queue):
        dict_pts = q.get()

        # Clear everything, seems not optimal
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # For now, let's not draw the camera
        self.d_cam.Activate(self.s_cam)

        # Draw the world frace
        pangolin.glDrawAxis(1)

        if dict_pts["est_pose"] is not None:
            gl.glColor3f(1., 0., 0.)
            pt = dict_pts["est_pose"]
            pangolin.glDrawPoints(pt)
        
        if dict_pts["gt_poses"] is not None:
            gl.glColor3f(0., 1., 0.)
            pt = dict_pts["gt_poses"]
            pangolin.glDrawPoints(pt)
        
        if dict_pts["cam_pts"] is not None:
            pt = dict_pts["cam_pts"]
            color = dict_pts["colors"]
            for i in range(len(pt)):
                gl.glColor3f(color[i]/255, color[i]/255, color[i]/255)
                pangolin.glDrawPoints([pt[i]])

        # Draw reference frames
        pangolin.FinishFrame()