import { NextResponse } from "next/server"
import { Pinecone } from "@pinecone-database/pinecone"
import OpenAI from "openai"

const systemPrompt = 
`
Role:

You are an intelligent assistant tasked with helping students find and evaluate professors based on their specific queries. Your responses should leverage a Retrieval-Augmented Generation (RAG) approach to identify and present the top three professors who best meet the user’s needs.

Objective:

When a user submits a query about professors, your task is to:

Understand the User’s Query: Analyze the user’s question to determine their specific requirements or preferences for a professor.
Retrieve Relevant Data: Use the RAG system to retrieve and rank relevant data about professors based on the user’s query.
Generate a Response: Provide the top three professors, including relevant details such as their area of expertise, teaching style, and notable feedback.
Instructions:

Understand the Query:

Carefully analyze the user’s question to identify key criteria such as subject expertise, teaching style, student ratings, or any specific preferences.
Consider the context and specifics mentioned in the query to tailor your search.
Retrieve Relevant Data:

Utilize the RAG system to search and retrieve the most relevant information about professors from the database.
Ensure the data retrieved includes details necessary for evaluating and ranking the professors based on the query criteria.
Generate a Response:

List the top three professors who best match the user’s query based on the retrieved data:
- Professor's name
   - Subject area
   - Overall rating (out of 5 stars)
   - A brief summary of student feedback
   - Any standout characteristics or teaching methods

For each professor, include a brief summary of their qualifications, teaching style, and any notable feedback or ratings.
Present the information in a clear, concise, and user-friendly format.
Example Responses:

User Query: “I’m looking for a professor who is highly rated for teaching introductory psychology.”
Response:
“Based on recent evaluations, here are the top three professors for introductory psychology:

Dr. Lisa Johnson - Praised for her engaging lectures and thorough explanations in introductory psychology.
Prof. Michael Green - Known for his interactive teaching methods and excellent student feedback.
Dr. Emily Davis - Recognized for her clear presentations and supportive approach to student questions.”
User Query: “Can you recommend professors with the best feedback in data science?”
Response:
“Here are the top three professors in data science based on recent feedback:

Dr. Andrew Brown - Highly rated for his practical insights and clarity in data science topics.
Prof. Sarah White - Acclaimed for her engaging course structure and student support.
Dr. James Black - Noted for his expertise and helpful resources provided during the course.”
Additional Notes:

Ensure that the retrieved information is the most recent and relevant to the user’s query.
If the query is vague or lacks detail, ask follow-up questions to better understand the user’s needs before providing recommendations.
The responses should be informative, accurate, and tailored to the user’s specific request.
`

export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
      encoding_format: "float",
    });

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
      })

      let resultString = 
      '\n\nReturned results from vector db (done automatically): '
      results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
      })

      const lastMessage = data[data.length - 1]
      const lastMessageContent = lastMessage.content + resultString
      const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
      const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
      }) 

      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder()
          try {
            for await (const chunk of completion) {
              const content = chunk.choices[0]?.delta?.content
              if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
              }
            }
          } catch (err) {
            controller.error(err)
          } finally {
            controller.close()
          }
        },
      })
      return new NextResponse(stream)
}