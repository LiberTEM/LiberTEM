import * as React from "react";

interface ScanSizePartProps {
    name: string,
    id: string,
    scanKey: number,
    value: number,
    scanSizeChangeHandle: (idx: number, value: string) => void,
    commaPressHandle: (idx: number) => void,
    scanRef: (ref:HTMLInputElement) => void,
}

const ScanSizePart: React.FC<ScanSizePartProps> = ({ name, id, scanKey, scanSizeChangeHandle, value, commaPressHandle, scanRef }) => {

   const onPartChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      scanSizeChangeHandle(scanKey, e.target.value);
   }

   const onCommaPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
     if(e.keyCode === 188) {
       commaPressHandle(scanKey);
     }
   }

   return <input type="number" name={name} id={id} key={scanKey} onChange={onPartChange} value={value} onKeyDown={onCommaPress} ref={scanRef} />;
}

export default ScanSizePart;
