import { useState } from 'react';
import { Loader2, BarChart3, Grid } from 'lucide-react';
import ImageUpload from '../components/ImageUpload';
import ResultsCard from '../components/ResultsCard';
import Disclaimer from '../components/Disclaimer';
import skinConditions from '../data/skinConditions.json';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function normalize(value = '') {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '');
}

export default function Scan() {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setAnalyzing(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Model prediction failed.');
      }

      if (!data.prediction || typeof data.confidence !== 'number') {
        throw new Error('The model returned an invalid response.');
      }

      const condition = skinConditions.find(item =>
        normalize(item.name) === normalize(data.prediction) ||
        normalize(item.id) === normalize(data.prediction)
      );

      setResult({
        title: data.prediction,
        confidence: data.confidence,
        topPredictions: data.top_predictions || [],
        overallMetrics: data.overall_metrics || {},
        classMetrics: data.class_metrics || {},
        description: condition
          ? `The model predicted ${data.prediction}. This is an AI-generated prediction, not a medical diagnosis.`
          : 'This is the disease class predicted by the trained model. Detailed information for this class is not available in the local condition database.',
        symptoms: condition?.symptoms || [],
        causes: condition?.causes || [],
        whenToSeeDoctor: condition?.whenToSeeDoctor,
        careTips: condition?.careTips || []
      });
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to the prediction server. Start the FastAPI backend and check VITE_API_URL.'
          : err.message || 'Analysis failed. Please try again.'
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const cm = result?.classMetrics?.confusion_matrix;

  return (
    <main className="min-h-[calc(100vh-8rem)] max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="font-poppins font-bold text-2xl md:text-3xl text-darkBrown text-center mb-2">
        Scan Your Skin
      </h1>

      <p className="text-warmGray text-center mb-8">
        Upload a photo for AI-based skin image classification. This is not a medical diagnosis.
      </p>

      <Disclaimer className="mb-8" />

      <div className="space-y-6">
        <ImageUpload
          onImageSelect={handleImageSelect}
          selectedFile={file}
          onClear={handleClear}
        />

        {file && (
          <div className="flex justify-center">
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={analyzing}
              className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-soft-lg bg-pink-accent text-white font-semibold shadow-soft hover:bg-pink disabled:opacity-70 disabled:cursor-not-allowed transition-all text-base min-h-[48px]"
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Running model...
                </>
              ) : (
                'Analyze'
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-soft-lg bg-red-50 border border-red-200 text-red-800 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-6 animate-in fade-in duration-300">
            {/* Primary Prediction & Top-3 Differential */}
            <div className="rounded-soft-lg bg-white border border-beige-300/50 shadow-soft p-5 md:p-6">
              <p className="text-sm text-warmGray">Model prediction</p>
              <h2 className="text-2xl font-poppins font-semibold text-darkBrown mt-1 capitalize">
                {result.title}
              </h2>
              <p className="text-sm text-warmGray mt-1">
                Confidence Score: <span className="font-semibold text-darkBrown">{result.confidence.toFixed(2)}%</span>
              </p>

              {result.topPredictions.length > 1 && (
                <div className="mt-4 pt-4 border-t border-beige-300/50">
                  <p className="text-xs font-semibold uppercase tracking-wider text-warmGray mb-2">
                    Top-3 Differential Probabilities
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {result.topPredictions.map((item, idx) => (
                      <div key={idx} className="p-3 rounded-soft bg-beige-200/50 border border-beige-300/50">
                        <p className="text-xs text-warmGray">#{idx + 1} Match</p>
                        <p className="text-sm font-semibold text-darkBrown capitalize truncate">{item.label}</p>
                        <p className="text-xs font-medium text-pink-accent mt-0.5">{item.confidence.toFixed(2)}%</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* F1 Score & Confusion Matrix Evaluation Card */}
            {(result.classMetrics?.f1_score !== undefined || result.overallMetrics?.weighted_f1 !== undefined) && (
              <div className="rounded-soft-lg bg-white border border-beige-300/50 shadow-soft p-5 md:p-6 space-y-6">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-pink-accent" />
                  <h3 className="text-lg font-poppins font-semibold text-darkBrown">
                    Model Evaluation Metrics ({result.title})
                  </h3>
                </div>

                {/* Metric Pills */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-soft bg-pink-light/30 border border-pink/30">
                    <p className="text-xs text-warmGray">Class F1 Score</p>
                    <p className="text-xl font-bold text-darkBrown mt-1">
                      {result.classMetrics.f1_score ?? 'N/A'}%
                    </p>
                  </div>
                  <div className="p-3.5 rounded-soft bg-beige-200/50 border border-beige-300/50">
                    <p className="text-xs text-warmGray">Class Precision</p>
                    <p className="text-xl font-bold text-darkBrown mt-1">
                      {result.classMetrics.precision ?? 'N/A'}%
                    </p>
                  </div>
                  <div className="p-3.5 rounded-soft bg-beige-200/50 border border-beige-300/50">
                    <p className="text-xs text-warmGray">Class Recall</p>
                    <p className="text-xl font-bold text-darkBrown mt-1">
                      {result.classMetrics.recall ?? 'N/A'}%
                    </p>
                  </div>
                  <div className="p-3.5 rounded-soft bg-beige-200/50 border border-beige-300/50">
                    <p className="text-xs text-warmGray">Overall Weighted F1</p>
                    <p className="text-xl font-bold text-darkBrown mt-1">
                      {result.overallMetrics.weighted_f1 ?? 'N/A'}%
                    </p>
                  </div>
                </div>

                {/* 2x2 One-vs-Rest Confusion Matrix */}
                {cm && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-darkBrown flex items-center gap-1.5">
                        <Grid className="w-4 h-4 text-pink-accent" />
                        2×2 One-vs-Rest Confusion Matrix (Validation Set)
                      </h4>
                      <span className="text-xs text-warmGray">
                        Class Support: {result.classMetrics.support} images
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse border border-beige-300 rounded-soft overflow-hidden">
                        <thead>
                          <tr className="bg-beige-200 text-darkBrown">
                            <th className="p-3 border border-beige-300 text-left">Actual \ Predicted</th>
                            <th className="p-3 border border-beige-300 text-center capitalize">
                              Predicted: {result.title}
                            </th>
                            <th className="p-3 border border-beige-300 text-center">
                              Predicted: Other Classes
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="p-3 border border-beige-300 font-semibold text-darkBrown bg-beige-200/40 capitalize">
                              Actual: {result.title}
                            </td>
                            <td className="p-3 border border-beige-300 text-center bg-green-50 text-green-900 font-bold">
                              {cm.tp} <span className="block text-xs font-normal text-green-700">True Positive (TP)</span>
                            </td>
                            <td className="p-3 border border-beige-300 text-center bg-red-50 text-red-900 font-semibold">
                              {cm.fn} <span className="block text-xs font-normal text-red-700">False Negative (FN)</span>
                            </td>
                          </tr>
                          <tr>
                            <td className="p-3 border border-beige-300 font-semibold text-darkBrown bg-beige-200/40">
                              Actual: Other Classes
                            </td>
                            <td className="p-3 border border-beige-300 text-center bg-amber-50 text-amber-900 font-semibold">
                              {cm.fp} <span className="block text-xs font-normal text-amber-700">False Positive (FP)</span>
                            </td>
                            <td className="p-3 border border-beige-300 text-center bg-beige-100/40 text-darkBrown font-semibold">
                              {cm.tn} <span className="block text-xs font-normal text-warmGray">True Negative (TN)</span>
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    {result.classMetrics.top_confused_with?.length > 0 && (
                      <p className="text-xs text-warmGray pt-1">
                        <strong>Most frequently confused with:</strong>{' '}
                        {result.classMetrics.top_confused_with
                          .map((c) => `${c.label} (${c.count} cases)`)
                          .join(', ')}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            <ResultsCard
              title={result.title}
              description={result.description}
              symptoms={result.symptoms}
              causes={result.causes}
              whenToSeeDoctor={result.whenToSeeDoctor}
              careTips={result.careTips}
              isAnalysisResult
            />

            <p className="text-center font-medium text-darkBrown">
              Consult a qualified dermatologist for diagnosis and treatment.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}