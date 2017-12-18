#pragma once

#ifndef __FACE_QUEUE_H
#define __FACE_QUEUE_H

#include <queue>
#include <mutex>

#define DEFAULT_QUEUE_SIZE 2

namespace OpenFace
{
/// <summary>
/// Simple thread-safe queue.
/// Wraps around std::queue and uses std::mutex for locking.
/// </summary>
template <typename T>
class Queue
{
private:
	::std::queue<T> mQueue;
	::std::mutex    mMutex;
	unsigned int    mMaxSize;

public:
	/// <summary>
	/// Constructor
	/// </summary>
	Queue(const unsigned int size = DEFAULT_QUEUE_SIZE) : mMaxSize(size) { }
	
	/// <summary>
	/// Destructor
	/// </summary>
	~Queue() { }

	/// <summary>
	/// Tries to enqueue an item.
	/// </summary>
	inline bool enqueue(T item)
	{
		bool ret = false;
		
		// lock
		mMutex.lock();
		
		// do
		if (mQueue.size() < mMaxSize)
		{
			ret = true;
			mQueue.push(item);
		}

		// unlock
		mMutex.unlock();
		
		return ret;
	}

	/// <summary>
	/// Tries to dequeue an item.
	/// </summary>
	inline T dequeue()
	{
		T ret = NULL;

		// lock
		mMutex.lock();
		
		// do
		if (!mQueue.empty())
		{
			ret = mQueue.front();
			mQueue.pop();
		}

		// unlock
		mMutex.unlock();		

		return ret;
	}

	/// <summary>
	/// Checks if the queue is empty.
	/// </summary>
	inline bool empty()
	{
		bool ret = true;

		// lock
		mMutex.lock();

		// do
		ret = mQueue.empty();

		// unlock
		mMutex.unlock();

		return ret;
	}
};
}
#endif
