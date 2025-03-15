// app/info/page.tsx
import React from "react";
import confidenceData from "../../data/confidenceIntervals.json";

interface ConfidenceEntry {
  range: string;
  interpretation: string;
}

export default function InfoPage() {
  const positiveResults: ConfidenceEntry[] = confidenceData.positive;
  const negativeResults: ConfidenceEntry[] = confidenceData.negative;

  return (
    <div className="min-h-screen bg-neutral-300">
      <div className="max-w-5xl mx-auto bg-white p-8 rounded shadow">
        <h1 className="text-3xl font-bold text-center mb-8">
          Confidence Levels and Interpretations
        </h1>

        <section className="mt-8 border-t pt-4">
          <p className="text-sm text-gray-600">
            Disclaimer: This tool is provided for informational purposes only and is not a
            substitute for professional medical advice. It should never be used as the sole
            diagnostic tool for determining the presence or absence of a brain tumor. Always
            consult with a qualified healthcare provider for a comprehensive evaluation.
          </p>
        </section>

        <section className="mt-8 mb-10">
          <h2 className="text-2xl font-semibold mb-4">Positive Results</h2>
          <table className="w-full table-auto border-collapse">
            <thead>
              <tr className="bg-gray-200">
                <th className="border px-4 py-2">Confidence Range</th>
                <th className="border px-4 py-2">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {positiveResults.map((entry, index) => (
                <tr key={entry.range} className={index % 2 === 1 ? "bg-gray-50" : ""}>
                  <td className="border px-4 py-2">{entry.range}</td>
                  <td className="border px-4 py-2">{entry.interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">Negative Results</h2>
          <table className="w-full table-auto border-collapse">
            <thead>
              <tr className="bg-gray-200">
                <th className="border px-4 py-2">Confidence Range</th>
                <th className="border px-4 py-2">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {negativeResults.map((entry, index) => (
                <tr key={entry.range} className={index % 2 === 1 ? "bg-gray-50" : ""}>
                  <td className="border px-4 py-2">{entry.range}</td>
                  <td className="border px-4 py-2">{entry.interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
