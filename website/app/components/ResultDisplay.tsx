"use client";
import React from "react";

interface Result {
  classification: boolean;
  confidence: number;
  error?: string | null;
}

interface ResultDisplayProps {
  result: Result | null;
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  if (!result) return null;

  if(result.error) {
    return (
      <div className="mt-6 text-center">
        <p className="font-medium text-red-500">Something went wrong processing your image. Please make sure you are using a .png or .jpg and try again</p>
      </div>
    )
  }

  return (
    <div className="mt-6 text-center">
      <div className="flex justify-between">
        <p className="text-lg font-medium">
          Classification:{" "}
          <span className="font-bold">
            {result.classification ? "Positive" : "Negative"}
          </span>
        </p>
        <p className="text-lg font-medium">
          Confidence:{" "}
          <span className="font-bold">
            {(result.confidence * 100).toFixed(2)}%
          </span>
        </p>
      </div>
    </div>
  );
}
