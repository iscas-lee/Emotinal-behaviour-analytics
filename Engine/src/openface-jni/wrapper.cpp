#include <FaceSystem.h>

#include "../../engine/include/zlib/zlib.h"
#include "../../engine/include/libpng/png.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/ocl.hpp>
#include <tbb/tbb.h>

#include <jni.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

using namespace OpenFace;

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "OpenFace++", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "OpenFace++", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "OpenFace++", __VA_ARGS__))

////////////////////////////////////////////////////////////////////////////////////////////////
///                                    OPENFACE++                                            ///
////////////////////////////////////////////////////////////////////////////////////////////////

// path where assets are saved to and read from
#define BASEPATH_RESOURCES      "/sdcard/"  

// size limits
#define MINWIDTH       128         
#define MINHEIGHT      128
#define MAXWIDTH       1280
#define MAXHEIGHT      960

// instances and variables
static JavaVM*     mJavaVM     = NULL;  // a persistent reference to the JavaVM instance
static FaceSystem* mFaceSystem = NULL;  // the face system instance created in init()
static int         mWidth      = 640;   // default width, set in init()
static int         mHeight     = 480;   // default height, set in init()
static int         mOutputSize = 640 * 480 * 4;

// the 'Result' java class instance receiving results from native side
// this is populated with new values in Dequeue() calls
static jobject     mResultJava              = NULL;

// the available field ids of the 'Result' java class
static jfieldID    mFieldPixelData          = NULL;
static jfieldID    mFieldFaceDetected       = NULL;
static jfieldID    mFieldAttention          = NULL;
static jfieldID    mFieldEmotFloatHappy     = NULL;
static jfieldID    mFieldEmotFloatSad       = NULL;
static jfieldID    mFieldEmotFloatSurprised = NULL;
static jfieldID    mFieldEmotFloatAngry     = NULL;
static jfieldID    mFieldEmotFloatDisgusted = NULL;
static jfieldID    mFieldEmotFloatFeared    = NULL;
static jfieldID    mFieldEmotFloatNeutral   = NULL;
static jfieldID    mFieldPositionX          = NULL;
static jfieldID    mFieldPositionY          = NULL;
static jfieldID    mFieldPositionZ          = NULL;
static jfieldID    mFieldOrientationX       = NULL;
static jfieldID    mFieldOrientationY       = NULL;
static jfieldID    mFieldOrientationZ       = NULL;

static jobject     mFieldPixelDataObj      = NULL;
static jbyteArray  mFieldPixelDataArray    = NULL;

////////////////////////////////////////////////////////////////////////////////////////////////

// android complains at linktime if this is missing, huh?
void fix_linking()
{
	int k = gzputs(0, 0);
	int h = gzclose(0);
	char* j = gzgets(0, 0, 0);
	::tbb::task_scheduler_init::default_num_threads();
	::cv::Mat temp;
	::cv::cvtColor(temp, temp, 0, 0);
	::cv::imdecode(temp, 0);
	::cv::circle(temp, ::cv::Point(0, 0), 0, 0, 0, 0, 0);
	::cv::getOptimalDFTSize(0);
	::cv::Rodrigues(temp, temp, temp);
	::cv::Algorithm();
	::cv::solvePnP(0, 0, 0, 0, temp, temp, 0, 0);
	::cvLog(0, 0);
	::cvIntegral(0, 0, 0, 0);
	::cvCanny(0, 0, 0, 0, 0);
	::cvGetQuadrangleSubPix(0, 0, 0);
	png_infop a = png_create_info_struct(0);
	svm_load_model("");
	::tbb::concurrent_vector<float> gja;
}

// helper to write zip-compressed APK assets to readable path
void writeAssets(AAssetManager* mgr)
{
	AAssetDir* assetDir = AAssetManager_openDir(mgr, "");
	const char* filename = (const char*)NULL;
	while ((filename = AAssetDir_getNextFileName(assetDir)) != NULL) {
		AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_STREAMING);
		char buf[BUFSIZ];
		int nb_read = 0;
		string sfilename = filename;
		FILE* out = fopen(("/sdcard/" + sfilename).c_str(), "w");
		while ((nb_read = AAsset_read(asset, buf, BUFSIZ)) > 0)
			fwrite(buf, nb_read, 1, out);
		fclose(out);
		AAsset_close(asset);
	}
	AAssetDir_close(assetDir);
}

/***********************************************************************************************************************************************************************/
/********************************************************************    JNI API     ***********************************************************************************/
/***********************************************************************************************************************************************************************/
extern "C" 
{
	jint JNI_OnLoad(JavaVM* jvm, void* reserved)
	{
		LOGI("JNI starting up");

		// keep a global reference to the jvm
		// so we can get a JNIEnv* from anywhere (can't cache JNIEnv*)
		mJavaVM = jvm;

		// check OpenCV hardware acceleration
		const bool ophv = cv::ocl::haveOpenCL();
		const bool opcl = cv::ocl::useOpenCL();

		if (cv::useOptimized())    LOGI("OpenCV-Optimized:  Yes"); else LOGI("OpenCV-Optimized:  No");
		if (cv::ocl::haveOpenCL()) LOGI("OpenCV-HaveOpenCL: Yes"); else LOGI("OpenCV-HaveOpenCL: No");
		if (cv::ocl::useOpenCL())  LOGI("OpenCV-UseOpenCL:  Yes"); else LOGI("OpenCV-UseOpenCL:  No");

		// define jni interface version
		return JNI_VERSION_1_6;
	}

	void JNI_OnUnload(JavaVM* jvm, void* reserved)
	{
		if (mFaceSystem)
			delete mFaceSystem;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	jboolean Java_com_example_test_openfaceandroid_Wrapper_Init(JNIEnv* env, jobject thiz, jint width, jint height, jobject result, jobject assetManager)
	{
		// can't init twice
		if (mFaceSystem)
		{
			LOGE("Already initialized. Can't init twice.");
			return false;
		}
			
		// verify sizes
		if (width < MINWIDTH || width > MAXWIDTH || height < MINHEIGHT || height > MAXHEIGHT)
		{
			LOGE("Width or Height out of bound.");
			return false;
		}

		// store them
		mWidth      = width;
		mHeight     = height;
		mOutputSize = width * height * 4;

		////////////////////////////////////////////////////////////////////////

		// turn java result instance into a global ref so we can keep it
		mResultJava = env->NewGlobalRef(result);
		if (!mResultJava)
		{
			LOGE("Failed to convert 'Result' instance into a global ref.");
			return false;
		}
	
		// get the java class for the result object to lookup fields
		const jclass RESULTCLASS = env->GetObjectClass(mResultJava);
		if (!RESULTCLASS)
		{
			LOGE("Failed to retrieve class of 'Result' instance.");
			return false;
		}

		////////////////////////////////////////////////////////////////////////

		// get field ids of result class
		mFieldPixelData          = env->GetFieldID(RESULTCLASS, "pixeldata",          "[B"); // byte[]
		mFieldFaceDetected       = env->GetFieldID(RESULTCLASS, "faceDetected",       "Z" ); // boolean
		mFieldAttention          = env->GetFieldID(RESULTCLASS, "attention",          "F" ); // float
		mFieldEmotFloatHappy     = env->GetFieldID(RESULTCLASS, "emotFloatHappy",     "F" ); // float
		mFieldEmotFloatSad       = env->GetFieldID(RESULTCLASS, "emotFloatSad",       "F" ); // float
		mFieldEmotFloatSurprised = env->GetFieldID(RESULTCLASS, "emotFloatSurprised", "F" ); // float
		mFieldEmotFloatAngry     = env->GetFieldID(RESULTCLASS, "emotFloatAngry",     "F" ); // float
		mFieldEmotFloatDisgusted = env->GetFieldID(RESULTCLASS, "emotFloatDisgusted", "F" ); // float
		mFieldEmotFloatFeared    = env->GetFieldID(RESULTCLASS, "emotFloatFeared",    "F" ); // float
		mFieldEmotFloatNeutral   = env->GetFieldID(RESULTCLASS, "emotFloatNeutral",   "F" ); // float
		mFieldPositionX          = env->GetFieldID(RESULTCLASS, "positionX",          "F" ); // float
		mFieldPositionY          = env->GetFieldID(RESULTCLASS, "positionY",          "F" ); // float
		mFieldPositionZ          = env->GetFieldID(RESULTCLASS, "positionZ",          "F" ); // float
		mFieldOrientationX       = env->GetFieldID(RESULTCLASS, "orientationX",       "F" ); // float
		mFieldOrientationY       = env->GetFieldID(RESULTCLASS, "orientationY",       "F" ); // float
		mFieldOrientationZ       = env->GetFieldID(RESULTCLASS, "orientationZ",       "F" ); // float

		// validate that fields were actually all found
		if (mFieldPixelData == NULL)          { LOGE("Missing field 'pixelData' on 'Result' class");          return false; }
		if (mFieldFaceDetected == NULL)       { LOGE("Missing field 'faceDetected' on 'Result' class");       return false; }
		if (mFieldAttention == NULL)          { LOGE("Missing field 'attention' on 'Result' class");          return false; }
		if (mFieldEmotFloatHappy == NULL)     { LOGE("Missing field 'emotFloatHappy' on 'Result' class");     return false; }
		if (mFieldEmotFloatSad == NULL)       { LOGE("Missing field 'emotFloatSad' on 'Result' class");       return false; }
		if (mFieldEmotFloatSurprised == NULL) { LOGE("Missing field 'emotFloatSurprised' on 'Result' class"); return false; }
		if (mFieldEmotFloatAngry == NULL)     { LOGE("Missing field 'emotFloatAngry' on 'Result' class");     return false; }
		if (mFieldEmotFloatDisgusted == NULL) { LOGE("Missing field 'emotFloatDisgusted' on 'Result' class"); return false; }
		if (mFieldEmotFloatFeared == NULL)    { LOGE("Missing field 'emotFloatFeared' on 'Result' class");    return false; }
		if (mFieldEmotFloatNeutral == NULL)   { LOGE("Missing field 'emotFloatNeutral' on 'Result' class");   return false; }
		if (mFieldPositionX == NULL)          { LOGE("Missing field 'positionX' on 'Result' class");          return false; }
		if (mFieldPositionY == NULL)          { LOGE("Missing field 'positionY' on 'Result' class");          return false; }
		if (mFieldPositionZ == NULL)          { LOGE("Missing field 'positionZ' on 'Result' class");          return false; }
		if (mFieldOrientationX == NULL)       { LOGE("Missing field 'orientationX' on 'Result' class");       return false; }
		if (mFieldOrientationY == NULL)       { LOGE("Missing field 'orientationY' on 'Result' class");       return false; }
		if (mFieldOrientationZ == NULL)       { LOGE("Missing field 'orientationZ' on 'Result' class");       return false; }

		////////////////////////////////////////////////////////////////////////

		// get pixeldata byte[] field of 'Result' instance
		const jobject ARROBJ = env->GetObjectField(mResultJava, mFieldPixelData);

		// turn java byte[] of 'Result' instance into a global ref so we can keep it
		mFieldPixelDataObj = env->NewGlobalRef(ARROBJ);
		if (!mFieldPixelDataObj)
		{
			LOGE("Failed to convert 'pixeldata' byte[] instance into a global ref.");
			return false;
		}
	
		// cast to native byte[]
		mFieldPixelDataArray = reinterpret_cast<jbyteArray>(mFieldPixelDataObj);

		// check size of managed byte[] to match outputsize
		const jsize SIZE = env->GetArrayLength(mFieldPixelDataArray);
		if (SIZE != mOutputSize)
		{
			LOGE("Invalid size of'pixeldata' byte[] in 'Result'. Allocate width*height*4!");
			return false;
		}
		
		////////////////////////////////////////////////////////////////////////
		// workaround: copy assets from zipped apk to readable filesystem
		LOGI("Extracting assets");
		AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
		writeAssets(mgr);
		
		////////////////////////////////////////////////////////////////////////
		// create openface++ system
		LOGI("Starting FaceSystem");
		mFaceSystem = new FaceSystem(BASEPATH_RESOURCES);
		mFaceSystem->setIsRunning(true);

		return true;
	}
	
	jboolean Java_com_example_test_openfaceandroid_Wrapper_Enqueue(JNIEnv* env, jobject thiz, jbyteArray dataArray)
	{
		if (!mFaceSystem)
			return false;

		// pin managed pixeldata so we can read it
		jbyte* data = env->GetByteArrayElements(dataArray, 0);
		
		// wrap opencv mat around pixels (must be NV21 YUV format!!)
		::cv::Mat mGray(mHeight + mHeight/2, mWidth, CV_8UC1, (unsigned char *)data);
		
		// must live on for workerthread, hence heap alloc
		::cv::Mat* mResult = new ::cv::Mat(mHeight, mWidth, CV_8UC4);

		// convert to something readable to openface++
		::cv::cvtColor(mGray, *mResult, CV_YUV2RGBA_NV21);
		
		// free pinned managed pixeldata
		env->ReleaseByteArrayElements(dataArray, (jbyte*)data, 0);
		
		// try to enqueue
		if (mFaceSystem->enqueueWork(mResult))
			return true;
		
		// cleanup heap alloc in case if failed
		else
		{
			delete mResult;
			return false;
		}
	}

	jboolean Java_com_example_test_openfaceandroid_Wrapper_Dequeue(JNIEnv* env, jobject thiz)
	{
		if (!mFaceSystem)
			return false;

		if (Result* result = mFaceSystem->dequeue())
		{			
			// copy data to managed output array
			env->SetByteArrayRegion(mFieldPixelDataArray, 0, mOutputSize, (jbyte*)result->image->data);
			
			// set misc values
			env->SetBooleanField(mResultJava, mFieldFaceDetected, result->faceDetected);

			// attention
			env->SetFloatField(mResultJava, mFieldAttention, (float)result->getAverageAttention());

			// set facial expression float regressor values
			env->SetFloatField(mResultJava, mFieldEmotFloatHappy,     (float)result->emotionProbability.happy);
			env->SetFloatField(mResultJava, mFieldEmotFloatSad,       (float)result->emotionProbability.sad);
			env->SetFloatField(mResultJava, mFieldEmotFloatSurprised, (float)result->emotionProbability.surprised);
			env->SetFloatField(mResultJava, mFieldEmotFloatAngry,     (float)result->emotionProbability.angry);
			env->SetFloatField(mResultJava, mFieldEmotFloatDisgusted, (float)result->emotionProbability.disgusted);
			env->SetFloatField(mResultJava, mFieldEmotFloatFeared,    (float)result->emotionProbability.feared);
			env->SetFloatField(mResultJava, mFieldEmotFloatNeutral,   (float)result->emotionProbability.neutral);

			// set face position
			env->SetFloatField(mResultJava, mFieldPositionX, (float)result->position[0]);
			env->SetFloatField(mResultJava, mFieldPositionY, (float)result->position[1]);
			env->SetFloatField(mResultJava, mFieldPositionZ, (float)result->position[2]);

			// set orientation
			env->SetFloatField(mResultJava, mFieldOrientationX, (float)result->orientation[0]);
			env->SetFloatField(mResultJava, mFieldOrientationY, (float)result->orientation[1]);
			env->SetFloatField(mResultJava, mFieldOrientationZ, (float)result->orientation[2]);

			// delete our native stuff
			delete result->image;
			delete result;

			// success
			return true;
		}

		// no result
		else
			return false;
	}

	jfloat Java_com_example_test_openfaceandroid_Wrapper_GetFPS(JNIEnv* env, jobject thiz)
	{
		if (!mFaceSystem)
			return 0.0f;

		return (float)mFaceSystem->getFPS();
	}
}
