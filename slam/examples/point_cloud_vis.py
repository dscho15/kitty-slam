import pypangolin as pango
import OpenGL.GL as gl
import numpy as np

def a_callback():
    print("a-pressed")

# 
pango.CreateWindowAndBind('ORB-SLAM', 1024, 768)

# 3D mouse handler requires depth testing to be enabled
gl.glEnable(gl.GL_DEPTH_TEST)

gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

# currently constant inputs to renderer
w, h, f = 640, 480, 2000

# http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
s_cam = pango.OpenGlRenderState(
    pango.ProjectionMatrix(w, h, f, f, w/2, h/2, 0.1, 1000),
    pango.ModelViewLookAt(-2, -2, -2, 0, 0, 0, pango.AxisDirection.AxisZ)
)

handler = pango.Handler3D(s_cam)

d_cam = (pango.CreateDisplay().SetBounds(pango.Attach(0.), pango.Attach(1.), pango.Attach.Pix(320), pango.Attach(1.), -1024/768).SetHandler(handler))

print("Went this far")
ctrl = -96
pango.RegisterKeyPressCallback(ctrl + ord("a"), a_callback)
print("Went this far")

trajectory = [[0, 0, 0]]
for i in range(10000):
    trajectory.append(trajectory[-1] + np.random.random(3)*0.25)
trajectory = np.array(trajectory)

# pango.OpenGlRenderState()


k = 0
while not pango.ShouldQuit():

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # print(d_cam.Activate(5))
        # #print(d_cam.Follow())
        # exit()

        if k % 100 == 20:

            s_cam = pango.OpenGlRenderState(
                pango.ProjectionMatrix(w, h, f, f, w/2, h/2, 0.1, 1000),
                pango.ModelViewLookAt(trajectory[k-10, 0], trajectory[k-10, 1], trajectory[k-10, 2], trajectory[k, 0], trajectory[k, 1], trajectory[k, 2], 0, 0, 1)
            )

            print(dir(s_cam.GetProjectionModelViewMatrix()))
            print(s_cam.GetProjectionModelViewMatrix().Matrix())

            print(trajectory[k, 0])

        d_cam.Activate(s_cam)
        print(s_cam.GetProjectionModelViewMatrix().Matrix())
        pango.glDrawAxis(1)
        pango.glDrawPoints(trajectory[:k])
        k += 1

        #pango.glDrawColouredCube()
        pango.FinishFrame()

        




# if __name__ == "__main__":
#     disp = Display()
