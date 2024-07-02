import React, { useState } from "react";

export default function Buttons({onRatingSubmit}) {
    const [rating, setRating] = useState(0);

    const handleRatingChange = (newRating) => {
        setRating(newRating);
      };

      const submitRating = () => {
        if (onRatingSubmit) {
          onRatingSubmit(rating);
        }
      };

    return (
        // <div>
        //     <button className="button-widget">1</button>
        //     <button className="button-widget">2</button>
        //     <button className="button-widget">3</button>
        //     <button className="button-widget">4</button>
        //     <button className="button-widget">5</button>
        // </div>

        <div className="rating-widget">
      <p className="title">Rate your experience:</p>
      <div className="rating-stars">
        {[1, 2, 3, 4, 5].map((value) => (
          <span
            key={value}
            className={`star ${value <= rating ? 'selected' : ''}`}
            onClick={() => handleRatingChange(value)}
          >
            ★
          </span>
        ))}
      </div>
      <button className="submitBtn" onClick={submitRating}>Submit</button>
    </div>
    );
}