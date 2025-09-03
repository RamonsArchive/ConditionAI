// Next.js API Route: pages/api/process-items.js
// or app/api/process-items/route.js (for App Router)

import { NextRequest, NextResponse } from 'next/server';

const CONDITIONAI_API_URL = process.env.CONDITIONAI_API_URL || 'http://localhost:8000';

export async function POST(request) {
  try {
    const body = await request.json();
    
    // Validate input
    if (!body.items || !Array.isArray(body.items)) {
      return NextResponse.json(
        { error: 'Items array is required' },
        { status: 400 }
      );
    }

    // Call the Python API
    const response = await fetch(`${CONDITIONAI_API_URL}/process-direct`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        items: body.items,
        max_items: body.max_items || null
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'API request failed');
    }

    const data = await response.json();
    
    return NextResponse.json({
      success: true,
      results: data.results,
      summary: data.summary
    });

  } catch (error) {
    console.error('Error processing items:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// For background processing (async jobs)
export async function POST_ASYNC(request) {
  try {
    const body = await request.json();
    
    // Start background job
    const response = await fetch(`${CONDITIONAI_API_URL}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        items: body.items,
        max_items: body.max_items || null
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'API request failed');
    }

    const data = await response.json();
    
    return NextResponse.json({
      success: true,
      job_id: data.job_id,
      message: data.message
    });

  } catch (error) {
    console.error('Error starting background job:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// Check job status
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('job_id');
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'job_id parameter is required' },
        { status: 400 }
      );
    }

    const response = await fetch(`${CONDITIONAI_API_URL}/job/${jobId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get job status');
    }

    const data = await response.json();
    
    return NextResponse.json({
      success: true,
      job: data
    });

  } catch (error) {
    console.error('Error getting job status:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}
