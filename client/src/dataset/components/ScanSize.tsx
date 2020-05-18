import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { parseNumListWithPadding } from "../helpers";
import ScanSizePart from "./ScanSizePart";

interface ScanSizeProps {
    value: string,
    minScan: number,
    maxScan: number,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
}

const ScanSize: React.FC<ScanSizeProps> = ({ value, minScan, maxScan, setFieldValue }) => {

    const scanSize = parseNumListWithPadding(value, minScan);
    
    const scanRefsArray = React.useRef<HTMLInputElement[]>([]);

    const scanSizeChangeHandle = (idx: number, val: string) => {
      const newScanSize = [...scanSize];
      newScanSize[idx] = val;
      setFieldValue("scan_size", newScanSize.toString());
    };

    const commaPressHandle = (idx: number) => {
      if(idx===(scanSize.length-1)) {
        newScanDim();
      } else {
          scanRefsArray.current[idx+1].focus();
      }
    }

    const newScanDim = () => {
      if(scanSize.length < maxScan) {
        const newScanSize = [...scanSize];
        newScanSize.push("");
        setFieldValue("scan_size", newScanSize.toString());
      }
    }

    React.useEffect(() => {
      if(scanSize.length > minScan) {
        scanRefsArray.current[scanSize.length-1].focus();
      }
    }, [scanSize.length, minScan]);

    const delScanDim = () => {
      if(scanSize.length > minScan) {
        const newScanSize = [...scanSize];
        newScanSize.pop();
        setFieldValue("scan_size", newScanSize.toString());
      }
    }

    return (
      <>
        <Form.Group>
          {scanSize.map((val, idx) => {
            const scanRef = (ref:HTMLInputElement) => { scanRefsArray.current[idx] = ref; }
            return <Form.Field width={2} key={idx}><ScanSizePart scanKey={idx} name={"scan_size_" + idx} id={"id_scan_size_" + idx} value={val} scanRef={scanRef} scanSizeChangeHandle={scanSizeChangeHandle} commaPressHandle={commaPressHandle} /></Form.Field>
          })}
          <Form.Field hidden={minScan === maxScan}>
            <Button onClick={newScanDim} disabled={scanSize.length === maxScan} type="button" icon="add" title="Add dimension" basic={false} />
            <Button onClick={delScanDim} disabled={scanSize.length === minScan} type="button" icon="minus" title="Remove dimension" basic={false} />
          </Form.Field>
        </Form.Group>
      </>
    );
}

export default ScanSize;
