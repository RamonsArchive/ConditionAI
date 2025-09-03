// React hook for ConditionAI integration
import { useState, useCallback } from 'react';

export const useConditionAI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const processItems = useCallback(async (items, maxItems = null) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('/api/process-items', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          items,
          max_items: maxItems
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process items');
      }

      const data = await response.json();
      setResults(data);
      return data;

    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const processItemsAsync = useCallback(async (items, maxItems = null) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/process-items-async', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          items,
          max_items: maxItems
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start processing');
      }

      const data = await response.json();
      return data.job_id;

    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const checkJobStatus = useCallback(async (jobId) => {
    try {
      const response = await fetch(`/api/process-items?job_id=${jobId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to check job status');
      }

      const data = await response.json();
      return data.job;

    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, []);

  return {
    loading,
    error,
    results,
    processItems,
    processItemsAsync,
    checkJobStatus,
    clearError: () => setError(null),
    clearResults: () => setResults(null)
  };
};

// Example React component
export const ConditionAIProcessor = ({ items, onResults }) => {
  const { loading, error, results, processItems } = useConditionAI();

  const handleProcess = async () => {
    try {
      const data = await processItems(items);
      onResults?.(data);
    } catch (err) {
      console.error('Processing failed:', err);
    }
  };

  return (
    <div className="condition-ai-processor">
      <button 
        onClick={handleProcess} 
        disabled={loading}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Processing...' : 'Analyze Conditions'}
      </button>
      
      {error && (
        <div className="text-red-500 mt-2">
          Error: {error}
        </div>
      )}
      
      {results && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold">Results</h3>
          <p>Processed {results.summary.total_items} items</p>
          {/* Render your results here */}
        </div>
      )}
    </div>
  );
};
