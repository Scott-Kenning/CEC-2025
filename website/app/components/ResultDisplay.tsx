"use client";
import React from "react";

interface Result {
  classification: string;
  confidence: number;
  error?: string | null;
  image?: string; // Base64-encoded PNG image of the heatmap
}

interface ResultDisplayProps {
  result: Result | null;
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  if (!result) return null;
  console.log(result);

  if(result.error) {
    return (
      <div className="mt-6 text-center">
        <p className="font-medium text-red-500">
          Something went wrong processing your image. Please make sure you are using a .png or .jpg and try again.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-6 text-center">
      <div className="flex flex-col items-center space-y-4">
        <div className="flex justify-between w-full max-w-md">
          <p className="text-lg font-medium">
            Classification:{" "}
            <span className="font-bold">{result.classification}</span>
          </p>
          <p className="text-lg font-medium">
            Confidence:{" "}
            <span className="font-bold">
              {(result.confidence * 100).toFixed(2)}%
            </span>
          </p>
        </div>
        {result.classification === "Positive" && result.image && (
          <div className="mt-4">
            <img
              src={`data:image/png;base64,${result.image}`}
              alt="Grad-CAM Heatmap"
              className="max-w-md rounded shadow"
            />
          </div>
        )}
      </div>
    </div>
  );
}
