import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function InputDialog({defaultURL, open, setOpen, submitCallback}) {
  const [url, setURL] = useState(defaultURL); 

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Enter Image URL</DialogTitle>
          <DialogDescription>
            Please provide a URL for the image you want to process
          </DialogDescription>
        </DialogHeader>
        <Input
          value={url}
          onChange={(e) => setURL(e.target.value)}
          placeholder="https://example.com/image.jpg"
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={()=>{submitCallback(url);setOpen(false)}}>Load Image</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}