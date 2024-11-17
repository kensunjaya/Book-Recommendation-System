
"use client";
import { useState } from "react";
import { Hourglass } from "react-loader-spinner";

interface Book {
  title: string;
  score: number;
  thumbnail: string;
  description: string;
  url: string;
}


export default function Home() {
  const [bookData, setBookData] = useState<Book[]>([]);
  const [loading, setLoading] = useState(false);
  const [title, setTitle] = useState("");
  const getRecommendation = async () => {
    if (title === "") {
      return;
    }
    try {
      setLoading(true);
      await fetch(`http://127.0.0.1:5000/recommend/${encodeURIComponent(title)}`)
        .then((res) => res.json())
        .then((data: Book[]) => {
          setBookData(data);
        });
    }
    catch (e) {
      console.error(e);
    }
    finally {
      setLoading(false);
    }
  };
  return (
    <main className="p-5">
      {loading && (
      <div className="fixed top-0 left-0 w-full h-full bg-black bg-opacity-40 z-50 flex items-center justify-center">
      <Hourglass
        visible={true}
        height="80"
        width="80"
        ariaLabel="hourglass-loading"
        wrapperStyle={{}}
        wrapperClass=""
        colors={['#000000', '#000000']}
      />
      </div>)}
      <div>
        <input type="text" placeholder="Enter book title ..." onChange={(e) => setTitle(e.target.value)} className="mr-5 border border-black px-3 py-1 rounded-sm min-w-[25rem]" />
        <button onClick={() => getRecommendation()} className="bg-black text-white py-1 px-3 hover:shadow-lg transition border border-black rounded-md">Get Recommendation</button>
        {bookData.map((book) => (
          <div key={book.title} className="my-5">
            <div className="font-semibold text-xl">{book.title}</div>
            <div className="font-mono">Distance Score: {book.score}</div>
            <div className="font-semibold">Description: </div>
            {book.description ? <div className="mb-5">{book.description}</div> : <div className="mb-5">No description available</div>}
            {book.thumbnail && <img src={book.thumbnail} alt={book.title} onClick={() => window.location.assign(book.url)} className="mb-3 hover:cursor-pointer hover:shadow-lg"/>}
            <hr className="border-1 border-gray-400"/>
          </div>
        ))}
      </div>
    </main>
  );
}
