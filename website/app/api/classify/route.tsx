import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file");

  if (!file || !(file instanceof Blob)) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  const externalApiUrl = "http://127.0.0.1:8000/classify";

  try {
    const externalResponse = await fetch(externalApiUrl, {
      method: "POST",
      body: formData,
    });

    console.log("External response:", externalResponse);
    const data = await externalResponse.json();
    console.log("Data:", data);

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error processing image:", error);
    return NextResponse.json(
      { error: "Error processing image" },
      { status: 500 }
    );
  }
}
