#pragma once

#ifndef __FACE_WORKER_H
#define __FACE_WORKER_H

#include <thread>
#include <atomic>
#include <chrono>
#include <FaceQueue.h>

#define WORKER_DEFAULT_QUEUESIZE  2

namespace OpenFace
{
/// <summary>
/// Base class wrapping an async worker-thread.
/// </summary>
template <typename T, typename U, typename W>
class Worker
{
protected:
	::std::thread                                          mThread;
	::std::atomic<bool>                                    mIsRunning;
	::std::atomic<float>                                   mFPS;
	::std::chrono::time_point<::std::chrono::high_resolution_clock> mTick;
	::std::chrono::time_point<::std::chrono::high_resolution_clock> mTickLast;
	::std::chrono::time_point<::std::chrono::high_resolution_clock> mTickLastMeasure;
	::std::chrono::duration<long long, ::std::nano>        mTickDelta;
	Queue<T>                                               mQueueIn;
	Queue<U>                                               mQueueOut;
	Queue<W>                                               mCommands;
	long long                                              mProcessedCount;

	virtual void init()             { }
	virtual void shutdown()         { }
	virtual void process(T item)    { }
	virtual void execute(W command) { }

	/// <summary>
	/// This code is executed by the internal worker-thread
	/// </summary>
	void threadProc()
	{
		// get init ticks
		mTickLast = ::std::chrono::high_resolution_clock::now();
		mTick = ::std::chrono::high_resolution_clock::now();

		// mini sleep
		::std::this_thread::sleep_for(::std::chrono::milliseconds(1));

		// init of subclasses
		init();

		///////////////////////////////////////////////////////////////////////////

		// threadloop
		while (mIsRunning.load())
		{
			// update current tick and delta to last tick
			mTick = ::std::chrono::high_resolution_clock::now();
			mTickDelta = mTick - mTickLast;

			///////////////////////////////////////////////////////////////////////

			// get span since last tps measuring
			const ::std::chrono::duration<long long, ::std::nano> NANOS_SINCE_MEASURE =
				mTick - mTickLastMeasure;

			// possibly calculate new TPS value for last span (every ~1000ms)
			if (NANOS_SINCE_MEASURE >= ::std::chrono::milliseconds(1000))
			{
				const float RATIO = (float)mProcessedCount / (float)NANOS_SINCE_MEASURE.count();
				mFPS.store(1000000000.0f * RATIO);

				// reset counter and store last TPS update tick
				mProcessedCount = 0;
				mTickLastMeasure = mTick;
			}

			///////////////////////////////////////////////////////////////////////

			// try get command
			W command = mCommands.dequeue();

			if (command)
			{
				execute(command);
			}

			///////////////////////////////////////////////////////////////////////

			// try get work item
			T item = mQueueIn.dequeue();

			// if there is an item, process it and immediately loop again
			if (item)
			{
				process(item);
				mProcessedCount++;
			}

			// otherwise sleep a bit
			else
				::std::this_thread::sleep_for(::std::chrono::milliseconds(1));

			///////////////////////////////////////////////////////////////////////

			// save this tick as last tick
			mTickLast = mTick;
		}

		// shutdown of subclasses
		shutdown();
	}

public:
	/// <summary>
	/// Constructor
	/// </summary>
	Worker() : 
		mIsRunning(false), 
		mQueueIn(WORKER_DEFAULT_QUEUESIZE),
		mQueueOut(WORKER_DEFAULT_QUEUESIZE) 
	{ }

	/// <summary>
	/// Destructor
	/// </summary>
	~Worker()
	{
		// free remaining queue items
		while (T item = mQueueIn.dequeue())
			delete item;
	}

	/// <summary>
	/// Starts or stops the worker
	/// </summary>
	void setIsRunning(bool isRunning)
	{
		if (!mIsRunning.load())
		{
			mIsRunning.store(true);
			mThread = ::std::thread(&Worker::threadProc, this);
			mThread.detach();
		}
		else
			mIsRunning.store(false);
	}

	/// <summary>
	/// Returns current FPS rate.
	/// </summary>
	__forceinline float getFPS() { return mFPS.load(); }

	/// <summary>
	/// Adds a new task for the worker.
	/// </summary>
	__forceinline bool enqueueWork(T item) { return mQueueIn.enqueue(item); }

	/// <summary>
	/// Returns the next task result from the worker.
	/// </summary>
	__forceinline U dequeueResult() { return mQueueOut.dequeue(); }
};
}
#endif
