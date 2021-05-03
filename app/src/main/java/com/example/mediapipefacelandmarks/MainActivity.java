package com.example.mediapipefacelandmarks;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.PixelFormat;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.graphics.SurfaceTexture;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;

import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.modules.facegeometry.FaceGeometryProto.FaceGeometry;
import com.google.mediapipe.formats.proto.MatrixDataProto.MatrixData;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends AppCompatActivity implements GLSurfaceView.Renderer {
    private static final String TAG = "MainActivity";
    private static final boolean FLIP_FRAMES_VERTICALLY = true;
    private static final int NUM_BUFFERS = 2;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        try {
            System.loadLibrary("opencv_java3");
        } catch (java.lang.UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
    }

    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    protected FrameProcessor processor;
    // Handles camera access via the {@link CameraX} Jetpack support library.
    protected CameraXPreviewHelper cameraHelper;

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;

    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;

    // ApplicationInfo for retrieving metadata defined in the manifest.
//    private ApplicationInfo applicationInfo;

    // Side packet / stream names.
    private static final String USE_FACE_DETECTION_INPUT_SOURCE_INPUT_SIDE_PACKET_NAME =
            "use_face_detection_input_source";
    private static final String SELECTED_EFFECT_ID_INPUT_STREAM_NAME = "selected_effect_id";
    private static final String OUTPUT_FACE_GEOMETRY_STREAM_NAME = "multi_face_geometry";

    private static final boolean USE_FACE_DETECTION_INPUT_SOURCE = false;

    private static final int SELECTED_EFFECT_ID_AXIS = 0;
    private static final int SELECTED_EFFECT_ID_GLASSES = 2;

    private final Object effectSelectionLock = new Object();
    private int selectedEffectId;

    private GLSurfaceView glView;
    private float x = 0.0f;
    private float y = 0.0f;
    private float z = 0.0f;

    private final ObjectRenderer andy = new ObjectRenderer();

    private static final float[] DEFAULT_COLOR = new float[]{0f, 0f, 0f, 0f};
    private float[] colorCorrectionRgba = new float[]{1.0f, 1.0f, 1.0f, 1.0f};
    float[] projectionMatrix = new float[16];
    private final float scaleFactor = 10.0f;
    private float[] viewMatrix = new float[16];
    private final int TARGET_LANDMARK = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();

        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this);

        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        "face_effect_gpu.binarypb",
                        "input_video",
                        "output_video");
        processor
                .getVideoSurfaceOutput()
                .setFlipY(true);

        PermissionHelper.checkAndRequestCameraPermissions(this);

        // By default, render the axis effect for the face detection input source and the glasses effect
        // for the face landmark input source.
        if (USE_FACE_DETECTION_INPUT_SOURCE) {
            selectedEffectId = SELECTED_EFFECT_ID_AXIS;
        } else {
            selectedEffectId = SELECTED_EFFECT_ID_GLASSES;
        }

        // Pass the USE_FACE_DETECTION_INPUT_SOURCE flag value as an input side packet into the graph.
        Map<String, Packet> inputSidePackets = new HashMap<>();
        inputSidePackets.put(
                USE_FACE_DETECTION_INPUT_SOURCE_INPUT_SIDE_PACKET_NAME,
                processor.getPacketCreator().createBool(USE_FACE_DETECTION_INPUT_SOURCE));
        processor.setInputSidePackets(inputSidePackets);

        // This callback demonstrates how the output face geometry packet can be obtained and used
        // in an Android app. As an example, the Z-translation component of the face pose transform
        // matrix is logged for each face being equal to the approximate distance away from the camera
        // in centimeters.
        processor.addPacketCallback(
                OUTPUT_FACE_GEOMETRY_STREAM_NAME,
                (packet) -> {
                    float[] modelMatrix = new float[16];
                    List<FaceGeometry> multiFaceGeometry =
                            PacketGetter.getProtoVector(packet, FaceGeometry.parser());

//                    Log.e(TAG, "Size of buffer list is " + multiFaceGeometry.get(0).getMesh().getVertexBufferList().size());
                    // Refer to mesh_3d.proto file
                    x = multiFaceGeometry.get(0).getMesh().getVertexBufferList().get(TARGET_LANDMARK*5 + 0);
                    y = multiFaceGeometry.get(0).getMesh().getVertexBufferList().get(TARGET_LANDMARK*5 + 1);
                    z = multiFaceGeometry.get(0).getMesh().getVertexBufferList().get(TARGET_LANDMARK*5 + 2);

                    MatrixData poseTransformMatrix = multiFaceGeometry.get(0).getPoseTransformMatrix();
                    modelMatrix[0] = poseTransformMatrix.getPackedData(0);
                    modelMatrix[1] = poseTransformMatrix.getPackedData(1);
                    modelMatrix[2] = poseTransformMatrix.getPackedData(2);
                    modelMatrix[3] = poseTransformMatrix.getPackedData(3);
                    modelMatrix[4] = poseTransformMatrix.getPackedData(4);
                    modelMatrix[5] = poseTransformMatrix.getPackedData(5);
                    modelMatrix[6] = poseTransformMatrix.getPackedData(6);
                    modelMatrix[7] = poseTransformMatrix.getPackedData(7);
                    modelMatrix[8] = poseTransformMatrix.getPackedData(8);
                    modelMatrix[9] = poseTransformMatrix.getPackedData(9);
                    modelMatrix[10] = poseTransformMatrix.getPackedData(10);
                    modelMatrix[11] = poseTransformMatrix.getPackedData(11);
                    modelMatrix[12] = poseTransformMatrix.getPackedData(12);
                    modelMatrix[13] = poseTransformMatrix.getPackedData(13);
                    modelMatrix[14] = poseTransformMatrix.getPackedData(14) + 8.0f;
                    modelMatrix[15] = poseTransformMatrix.getPackedData(15);
                    Matrix.translateM(modelMatrix, 0, x, y, z);
                    andy.updateModelMatrix(modelMatrix, scaleFactor);

//                    glView.requestRender();
                });

        // Alongside the input camera frame, we also send the `selected_effect_id` int32 packet to
        // indicate which effect should be rendered on this frame.
        processor.setOnWillAddFrameListener(
                (timestamp) -> {
                    Packet selectedEffectIdPacket = null;
                    try {
                        synchronized (effectSelectionLock) {
                            selectedEffectIdPacket = processor.getPacketCreator().createInt32(selectedEffectId);
                        }

                        processor
                                .getGraph()
                                .addPacketToInputStream(
                                        SELECTED_EFFECT_ID_INPUT_STREAM_NAME, selectedEffectIdPacket, timestamp);
                    } catch (RuntimeException e) {
                        Log.e(
                                TAG, "Exception while adding packet to input stream while switching effects: " + e);
                    } finally {
                        if (selectedEffectIdPacket != null) {
                            selectedEffectIdPacket.release();
                        }
                    }
                });
    }

    @Override
    protected void onResume() {
        super.onResume();
        converter =
                new ExternalTextureConverter(
                        eglManager.getContext(), NUM_BUFFERS);
        converter.setFlipY(true);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }

        if (glView != null) glView.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();

        // Hide preview display until we re-open the camera again.
        previewDisplayView.setVisibility(View.GONE);

        if (glView != null) glView.onPause();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        previewFrameTexture = surfaceTexture;
        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    protected Size cameraTargetResolution() {
        return null; // No preference and let the camera (helper) decide.
    }

    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });

        glView = new GLSurfaceView(this);
        glView.setPreserveEGLContextOnPause(true);
        glView.setEGLContextClientVersion(2);
        glView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        glView.getHolder().setFormat(PixelFormat.TRANSLUCENT); // Needs to be a translucent surface so the camera preview shows through.
        glView.setRenderer(this);
        glView.setWillNotDraw(false);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY); // Only render when we have a frame (must call requestRender()).
        glView.setZOrderMediaOverlay(true); // Request that GL view's SurfaceView be on top of other SurfaceViews (including CameraPreview's SurfaceView).
        addContentView(glView, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
    }

    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    private void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    onCameraStarted(surfaceTexture);
                });
        cameraHelper.startCamera(this, CameraHelper.CameraFacing.FRONT, /*surfaceTexture=*/ null, cameraTargetResolution());
    }

    protected void onPreviewDisplaySurfaceChanged(
            SurfaceHolder holder, int format, int width, int height) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        boolean isCameraRotated = cameraHelper.isCameraRotated();

        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
        float aspect_ratio = (float)width / (float)height;

        Log.d(TAG, "WIDTH IS " + width + " HEIGHT IS " + height);
        Matrix.perspectiveM(projectionMatrix, 0, 90, aspect_ratio, 1f, 10000f);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        gl.glClearColor(0f, 0f, 0f, 0f);
        // Default view matrix of the opengl: http://www.songho.ca/opengl/gl_transform.html
        viewMatrix[0] = 1;
        viewMatrix[1] = 0;
        viewMatrix[2] = 0;
        viewMatrix[3] = 0;
        viewMatrix[4] = 0;
        viewMatrix[5] = 1;
        viewMatrix[6] = 0;
        viewMatrix[7] = 0;
        viewMatrix[8] = 0;
        viewMatrix[9] = 0;
        viewMatrix[10] = 1;
        viewMatrix[11] = 0;
        viewMatrix[12] = 0;
        viewMatrix[13] = 0;
        viewMatrix[14] = 0;
        viewMatrix[15] = 1;

        try {
            andy.createOnGlThread(
                    /*context=*/ this,
                    "andy.obj",
                    "andy.png");
        } catch (IOException e) {
            e.printStackTrace();
        }
        andy.setMaterialProperties(0.0f, 1.0f, 0.1f, 6.0f);
        andy.setBlendMode(ObjectRenderer.BlendMode.AlphaBlending);
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        if (height == 0) height = 1;   // To prevent divide by zero
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        try{
            andy.draw(viewMatrix, projectionMatrix, colorCorrectionRgba, DEFAULT_COLOR);
            Log.d(TAG, "projectionMatrix is " + Arrays.toString(projectionMatrix));
            Log.d(TAG, "mViewMatrix is " + Arrays.toString(viewMatrix));
        } catch (Throwable throwable) {
            Log.e(TAG, "Throw aways...");
        }
    }
}