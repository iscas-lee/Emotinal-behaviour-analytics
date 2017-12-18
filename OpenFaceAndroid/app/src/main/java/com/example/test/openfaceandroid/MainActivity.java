package com.example.test.openfaceandroid;

import android.Manifest;
import android.annotation.TargetApi;
import android.os.Build;
import android.support.annotation.NonNull;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.Handler;
import android.os.Looper;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import java.nio.ByteBuffer;

import android.graphics.ImageFormat;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

/**
 *
 */
public final class MainActivity extends AppCompatActivity
{
    private final static int WIDTH       = 640;
    private final static int HEIGHT      = 480;
    private final static int PIXELBYTES  = 4;
    private final static int BYTESIZE    = WIDTH * HEIGHT * PIXELBYTES;
    private final static int RESULTSPAN  = 33;
    private final static int PERMISSION_REQUEST = 1;
    private final static Handler handler = new Handler(Looper.getMainLooper());
    private final static Bitmap bitmap = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888);
    private final static byte[][] buffers = new byte[2][BYTESIZE];

    private static boolean isPaused;
    private static boolean hasPermissions;

    private Camera camera;

    private View vContainer;
    private TextView vTitle;
    private ImageView vImageView;
    private ProgressBar vEmotFloatHappy;
    private ProgressBar vEmotFloatSad;
    private ProgressBar vEmotFloatSurprised;
    private ProgressBar vEmotFloatAngry;
    private ProgressBar vEmotFloatDisgusted;
    private ProgressBar vEmotFloatFeared;
    private ProgressBar vEmotFloatNeutral;
    private ProgressBar vAttention;
    private TextView vPosition;
    private TextView vOrientation;

    private final Runnable checkResult = new Runnable()
    {
        @Override
        public final void run()
        {
            if (Wrapper.dequeue() &&
                Wrapper.result.pixeldata != null &&
                Wrapper.result.pixeldata.length >= BYTESIZE)
            {
                final ByteBuffer bytebuf = ByteBuffer.wrap(Wrapper.result.pixeldata);

                // show last frame
                bitmap.copyPixelsFromBuffer(bytebuf);
                vImageView.setImageBitmap(bitmap);

                // set emotion svr values
                vEmotFloatHappy.setProgress((int)(Wrapper.result.emotFloatHappy * 100f));
                vEmotFloatSad.setProgress((int)(Wrapper.result.emotFloatSad * 100f));
                vEmotFloatSurprised.setProgress((int)(Wrapper.result.emotFloatSurprised * 100f));
                vEmotFloatAngry.setProgress((int)(Wrapper.result.emotFloatAngry * 100f));
                vEmotFloatDisgusted.setProgress((int)(Wrapper.result.emotFloatDisgusted * 100f));
                vEmotFloatFeared.setProgress((int)(Wrapper.result.emotFloatFeared * 100f));
                vEmotFloatNeutral.setProgress((int)(Wrapper.result.emotFloatNeutral * 100f));

                // set attention
                vAttention.setProgress((int)(Wrapper.result.attention * 100f));

                // set position
                vPosition.setText(
                    String.valueOf((int)Wrapper.result.positionX) + "mm / " +
                    String.valueOf((int)Wrapper.result.positionY) + "mm / " +
                    String.valueOf((int)Wrapper.result.positionZ) + "mm");

                final int degX = (int)((Wrapper.result.orientationX * 180.0f) / (float)Math.PI);
                final int degY = (int)((Wrapper.result.orientationY * 180.0f) / (float)Math.PI);
                final int degZ = (int)((Wrapper.result.orientationZ * 180.0f) / (float)Math.PI);

                // set orientation
                vOrientation.setText(
                    String.valueOf(degX) + "° / " +
                    String.valueOf(degY) + "° / " +
                    String.valueOf(degZ) + "°");
            }

            // update FPS rate in title
            vTitle.setText("OpenFace++ @ " + String.valueOf((int)Wrapper.getFPS()) + " FPS");

            // schedule next check
            // 33ms = ~30fps max output fps
            if (!isPaused)
                handler.postDelayed(checkResult, RESULTSPAN);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        // don't sleep/suspend while in player activity
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // load content
        setContentView(R.layout.activity_main);

        ////////////////////////////////////////////////////////////////////////////////////////////

        // get sub-controls
        vContainer = (View)      findViewById(R.id.container);
        vTitle     = (TextView)  findViewById(R.id.title);
        vImageView = (ImageView) findViewById(R.id.frame);

        // get emotion bars
        vEmotFloatHappy     = (ProgressBar) findViewById(R.id.emot_float_happy);
        vEmotFloatSad       = (ProgressBar) findViewById(R.id.emot_float_sad);
        vEmotFloatSurprised = (ProgressBar) findViewById(R.id.emot_float_surprised);
        vEmotFloatAngry     = (ProgressBar) findViewById(R.id.emot_float_angry);
        vEmotFloatDisgusted = (ProgressBar) findViewById(R.id.emot_float_disgusted);
        vEmotFloatFeared    = (ProgressBar) findViewById(R.id.emot_float_feared);
        vEmotFloatNeutral   = (ProgressBar) findViewById(R.id.emot_float_neutral);

        // get attention, position and orientation
        vAttention   = (ProgressBar) findViewById(R.id.attention);
        vPosition    = (TextView)    findViewById(R.id.position);
        vOrientation = (TextView)    findViewById(R.id.orientation);

        ////////////////////////////////////////////////////////////////////////////////////////////

        // request runtime permissions for Android 6+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPerm();
        }
        else
        {
            hasPermissions = true;

            // initialize openface++
            Wrapper.init(WIDTH, HEIGHT, getResources().getAssets());
        }
    }

    @TargetApi(23)
    private void requestPerm()
    {
        requestPermissions(new String[] {
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
            },
            PERMISSION_REQUEST);
    }

    @Override
    public final void onRequestPermissionsResult(
        final int requestCode,
        @NonNull final String permissions[],
        @NonNull final int[] grantResults)
    {
        switch (requestCode)
        {
            case PERMISSION_REQUEST:
                if (grantResults.length == 0)
                    return;

                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    hasPermissions = true;
                    initCamera();
                    Wrapper.init(WIDTH, HEIGHT, getResources().getAssets());
                }
                break;
        }
    }

    @Override
    protected void onPause()
    {
        super.onPause();

        if (camera != null) {
            camera.stopPreview();
            camera.setPreviewCallbackWithBuffer(null);
            camera.release();
            camera = null;
        }

        isPaused = true;
        handler.removeCallbacks(checkResult);
    }

    @Override
    protected void onResume()
    {
        super.onResume();

        // unmark paused
        isPaused = false;

        // go to fullscreen
        if (vContainer != null)
        {
            vContainer.setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LOW_PROFILE |
                View.SYSTEM_UI_FLAG_FULLSCREEN |
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE |
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        }

        handler.postDelayed(checkResult, RESULTSPAN);

        // reinit camera
        if (hasPermissions)
            initCamera();
    }

    private void initCamera()
    {
        // open and lock device
        camera = Camera.open();

        // set size and imageformat
        // beware: the native side expects YUV NV21 here!!
        Camera.Parameters params = camera.getParameters();
        params.setPreviewSize(WIDTH, HEIGHT);
        params.setPreviewFormat(ImageFormat.NV21);

        // apply params
        camera.setParameters(params);

        // add our own buffers to avoid a lot of gc
        for (int i = 0; i < buffers.length; i++)
            camera.addCallbackBuffer(buffers[i]);

        // register frame callback and start
        camera.setPreviewCallbackWithBuffer(new PreviewCallback());
        camera.startPreview();
    }

    /**
     * Invoked by the camera for each frame. Provides pixels as raw byte[].
     */
    private final static class PreviewCallback implements Camera.PreviewCallback
    {
        @Override
        public final void onPreviewFrame(final byte[] data, final Camera camera)
        {
            // forward pixeldata to native openface++
            Wrapper.enqueue(data);

            // re-add the just used buffer to the available ones
            camera.addCallbackBuffer(data);
        }
    }
}
